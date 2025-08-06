import torch 
import time
import zmq 
import json
from typing import List, Dict, Any, Optional, Tuple, Type

from .requests import Request 
from .worker import ModelWorker
from .sampling import SamplingConfig

class Scheduler:
    def __init__(
        self, 
        model_name_or_path: str,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 1,
        request_socket_path: str = "/tmp/vox_serve_request.ipc",
        result_socket_path: str = "/tmp/vox_serve_reqult.ipc"
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.model_worker = ModelWorker(model_name_or_path, max_batch_size=max_batch_size)

        self.active_requests: List[Request] = []

        self.context = zmq.Context()
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.bind(f"ipc://{request_socket_path}")
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.bind(f"ipc://{result_socket_path}")

    def _step(self):
        """
        Process the next batch of requests.
        """

        # insert/remove requests to self.active_requests 
        self._prepare_requests()

        # TODO: advanced scheduling logic here for LM forward
        requests = self.active_requests

        is_prefill = False 
        for req in requests:
            is_prefill = is_prefill or (not req.done_lm_prefill)

        if len(requests) == 0:
            time.sleep(0.1)
            return

        # run either prefill or decode of LM 
        if is_prefill:
            self.model_worker.run_lm_prefill(requests)
        else:
            self.model_worker.run_lm_decode(requests)

        # check for request completion and mark as done
        for req in requests:
            if self.model_worker.is_finished(req):
                req.done_all = True

        # TODO: advanced scheduling logic here for detokenization (independent of LM forward),
        # including multiple chunks from a single request for some cases
        requests_to_detokenize = []
        
        for req in requests:
            if req.done_all or (len(req.lm_output_tokens) - req.next_audio_decode_idx >= self.model_worker.detokenize_interval):
                requests_to_detokenize.append(req)
        
        # run detokenization if needed
        self.model_worker.run_detokenize(requests_to_detokenize)

        # return results to clients
        for req in requests:
            if not req.output_audio.empty():
                # Send audio chunk message: request_id|AUDIO|audio_data
                message = req.request_id.encode('utf-8') + b'|AUDIO|' + req.output_audio.get()
                self.result_socket.send(message)
            
            # send completion notification for finished requests
            if req.done_all:
                self.model_worker.free_kv_cache(req)
                completion_message = {
                    'status': 'completed',
                    'reason': 'position_limit_exceeded'
                }
                # Send completion message: request_id|COMPLETION|json_data
                completion_payload = (req.request_id.encode('utf-8') + b'|COMPLETION|' + 
                                    json.dumps(completion_message).encode('utf-8'))
                self.result_socket.send(completion_payload)
        
        return
    
    def run_forever(self):
        """
        Run the scheduler indefinitely.
        """
        while True:
            self._step()
            # Optionally, sleep or yield to avoid busy-waiting
            torch.cuda.synchronize()
    
    def _prepare_requests(self):
        """
        Prepare requests for processing.
        This method should be implemented to gather new requests from clients.
        TODO: make it async with _step() function
        """

        # get new requests from ZMQ
        while True:
            try:
                message_payload = self.request_socket.recv(flags=zmq.NOBLOCK)
                delimiter_pos = message_payload.find(b'|') 
                if delimiter_pos != -1:
                    # Parse JSON request data
                    json_data = message_payload[:delimiter_pos].decode('utf-8')
                    request_dict = json.loads(json_data)
                    
                    # Create Request object from deserialized data
                    new_request = Request(
                        request_id=request_dict['request_id'],
                        prompt=request_dict['prompt'],
                    )

                    print(f"{new_request=}")
                    
                    # Store voice information as attribute (not part of Request dataclass)
                    # new_request.voice = request_dict.get('voice', 'tara')
                    
                    self.active_requests.append(new_request)
                else:
                    print(f"[WARNING] Received malformed audio message: {message_payload[:50]}...")
            except zmq.Again:
                break
            except Exception as e:
                print(f"[ERROR] Error receiving requests: {str(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # Filter out completed requests
        self.active_requests = [req for req in self.active_requests if not req.done_all]
        
        return 
import torch 
import zmq 

from .requests import Request 
from .worker import ModelWorker

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

        self.active_requests = []

        self.context = zmq.Context()
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.bind(f"ipc://{request_socket_path}")
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.bind(f"ipc://{result_socket_path}")

    def _step(self):
        """
        Process the next batch of requests.
        """

        # get new requests 
        requests, is_prefill = self._prepare_requests()

        # run either prefill or decode of LM 
        if is_prefill:
            self.model_worker.run_lm_prefill(requests)
        else:
            self.model_worker.run_lm_decode(requests)

        # run detokenization if needed
        self.model_worker.run_detokenize(requests)

        # return results to clients
    
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
                    # TODO: actually implement here
                    new_request = message_payload.decode('utf-8')
                    self.active_requests.append(new_request)
                else:
                    print(f"[WARNING] Received malformed audio message: {message_payload[:50]}...")
            except zmq.Again:
                break
            except Exception as e:
                print(f"[ERROR] Error receiving requests: {str(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        is_prefill = False 
        for req in self.active_requests:
            is_prefill = is_prefill or (not req.done_lm_prefill)
        
        return self.active_requests, is_prefill
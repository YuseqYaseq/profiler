import sys
import time


class Profiler:
    def __init__(self):
        self.prev_trace = None
        self.timer_dict = {}

    def _trace(self, frame, event, arg):
        try:
            # This approach is pretty heavy unfortunately. We also don't do any callibration.
            if event.startswith('c_'):
                func_name = (arg.__module__ + '.' if arg.__module__ is not None else "") + arg.__name__
            else:
                func_name = frame.f_code.co_qualname

            if event in ['call', 'c_call']:
                if func_name not in self.timer_dict:
                    self.timer_dict[func_name] = [0.0, []]  # summed time; call stack to enable recurrent calls
                start = time.time()
                self.timer_dict[func_name][1].append(start)
            elif event in ['return', 'c_return', 'c_exception']:
                end = time.time()
                try:
                    start = self.timer_dict[func_name][1].pop()
                except KeyError:
                    return  # this is Profiler.__enter__; we can ignore it.
                elapsed = end - start
                self.timer_dict[func_name][0] += elapsed
            else:
                print(func_name)
                print(f"{frame}; {event}; {arg}")
        except Exception as e:
            print(frame, event, arg, e)
            raise e

    def __enter__(self):
        self.prev_trace = sys.getprofile()
        sys.setprofile(self._trace)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setprofile(self.prev_trace)

    def print_stats(self, top_k=20):
        values = [(name, value[0]) for name, value in self.timer_dict.items()]
        values.sort(key=lambda x: x[1], reverse=True)
        print(f"Name | Time total")
        for name, value in values[:top_k]:
            print(f"{name}: {value*1000:0.3f}ms")

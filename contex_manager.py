class Identer(): 
    def __init__(self): 
        self.count_tabs = -1
        pass
          
    def __enter__(self): 
        self.count_tabs += 1 
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        pass
    
    def print(self, phrase):
        out_str = ''
        for i in range(self.count_tabs):
            out_str += '\t'
        print(out_str + phrase)
  
  
with Identer() as manager: 
    manager.print("hi")
    with manager as m:
        m.print("hello")
        with m as m:
            m.print('bonj')
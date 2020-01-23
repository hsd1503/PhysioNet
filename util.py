def get_label_map(i):
    """
    Association for the Advancement of Medical Instrumentation. "ANSI/AAMI EC57: 2012—Testing and Reporting Performance Results of Cardiac Rhythm and ST Segment Measurement Algorithms." American National Standard (2013).
    """
    m = {'N':'N', 
         'L':'N', 
         'R':'N', 
         'V':'V', 
         '/':'Q', 
         'A':'S', 
         'F':'F', 
         'f':'Q', 
         'j':'S', 
         'a':'S', 
         'E':'V', 
         'J':'S', 
         'e':'S', 
         'Q':'Q', 
         'S':'S'}
    return m[i]
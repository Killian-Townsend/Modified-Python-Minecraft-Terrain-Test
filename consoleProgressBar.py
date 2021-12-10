import sys

def startProgress(title):
    global progress_x
    sys.stdout.write("[" + "-"*50 + "]")
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    progress_x = int(x * 50 // 100)
    sys.stdout.write("["+("-"*50)+"]"+" : "+str(x)+"%")
    sys.stdout.write("\r["+("="*(progress_x-1))+">")
    sys.stdout.flush()

def endProgress():
    sys.stdout.write("\n[" + "="*50 + "]")
    sys.stdout.flush()
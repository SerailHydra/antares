# the dag object

import ptx
from ptx import *

class dag:
    def __init__(self, ptx_file):
        print(ptx_file)
        self.params = []
        self.regs = {} # name -> {dtype, val}
        self.asms = []
        self.labels = {} # label -> line
        with open(ptx_file, "r") as f:
            # init params
            lines = f.readlines()
            cur = 0
            while cur < len(lines):
                line = lines[cur].strip()
                cur += 1
                args = line.split(" ")
                if args[0] == ".param":
                    self.params.append(args[-1])
                    if self.params[-1][-1] == ",":
                        self.params[-1] = self.params[-1][:-2]
                if args[0] == "{":
                    break
            # init regs
            while cur < len(lines):
                line = lines[cur].strip()
                cur += 1
                args = line.split(" ")
                if args[0] == ".reg":
                    i = args[2].find("<")
                    j = args[2].find(">")
                    rlen = int(args[2][i+1:j])
                    name = args[2][:i].strip()
                    for k in range(rlen):
                        rname = name + str(k+1)
                        self.regs[rname] = {"dtype": args[1], "val": None}
                else:
                    break
            cur += 1
            # init ptx
            while cur < len(lines):
                line = lines[cur].strip()
                cur += 1
                args = line.split("\t")
                asm = ptx(args)
                if asm.cmd == "label":
                    self.labels[asm.name] = len(self.asms)
                elif asm.cmd:
                    self.asms.append(asm)
        # set block indecies to unknown
        self.regs["%ctaid.x"] = {"val": None}
        self.regs["%ctaid.y"] = {"val": None}

    def _build_dag(self):
        
        pass

    def simulate(self):
        GLSProgress = 0
        SASProgress = 0
        CSProgress = 0

        for rname in self.regs:
            self.regs[rname]["val"] = None

        asm_i = 0
        while asm_i != -1:
            asm = self.asms[asm_i]
            asm_i = asm.execute(self.regs, self.labels, asm_i)
            



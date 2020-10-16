# the ptx object

class ptx():

    def __init__(self, args):
        self.cmd = None
        if len(args) == 1:
            # is empty
            if len(args[0]) == 0:
                return
            # is label
            if args[0][-1] == ":":
                self.cmd = "label"
                self.name = args[0][:-1]
            if args[0].startswith("ret"):
                self.cmd = "ret"
            return
        # branch
        if "bra" in args[0]:
            self.cmd = "bra"
            i = args[0].find(" ")
            self.cond = args[0][1:i]
            self.dst = args[1][:-1]
            return
        # normal asms
        self.cmd = args[0]
        params = args[1].split(", ")
        # remove ;
        params[-1] = params[-1][:-1]
        self.inputs = params[1:]
        self.outputs = params[0]
        print(self.cmd, params)


    def execute(self, regs, labels, asm_i):
        # load and store: pass
        if self.cmd.startswith("ld.") or self.cmd.startswith("st."):
            return asm_i + 1
        # return
        if self.cmd == "ret":
            return -1
        # branching 
        if self.cmd == "bra":
            assert regs[self.cond]["val"] != None
            if regs[self.cond]["val"]:
                return labels[self.dst]
            else:
                return asm_i + 1
        # compute: calculate the result if registers are known
        flag = True
        v = []
        for in_reg in self.inputs:
            if in_reg.startswith("%") and regs[in_reg]["val"] == None:
                # unknown registers
                flag = False
            elif in_reg.startswith("%"):
                # is a register
                v.append(regs[in_reg]["val"])
            elif "f" in in_reg or "x" in in_reg:
                # is a hex immediate
                v.append(int(in_reg, 16))
            else:
                # is an immediate
                v.append(int(in_reg))
        if not flag:
            # skip when there are unknown registers
            regs[self.outputs]["val"] = None
            return asm_i + 1
        if self.cmd.startswith("setp.ne."):
            regs[self.outputs]["val"] = v[0] != v[1]
        if self.cmd.startswith("mov."):
            regs[self.outputs]["val"] = v[0]
        if self.cmd.startswith("fma."):
            regs[self.outputs]["val"] = v[0] * v[1] + v[2]
        if self.cmd.startswith("add."):
            regs[self.outputs]["val"] = v[0] + v[1]
        return asm_i + 1

    def exec_stream(self):
        if self.cmd.startswith("ld.") or self.cmd.startswith("st."):
            return "GLS"
        if self.cmd.startswith("sts."):
            return "SAS"
        return "CS"

    def latency(self):
        if exec_stream() == "GLS":
            # total load / DRAM bandwidth
            pass
        if exec_stream() == "CS":
            pass


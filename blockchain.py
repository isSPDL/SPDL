# coding:utf-8
import grpc
from ecdsa import SigningKey, NIST384p, VerifyingKey
import grpc_pb2
import grpc_pb2_grpc
import time
import re
import json
import hashlib
import threading
import p2p

_compiNum = re.compile("^\d+$")  # 判斷全數字用
_compiW = re.compile("^\w{64}")

# TODO
with open('svkey.json', 'r', encoding='utf8')as fp:
    svkey = json.load(fp)


def signing(privatekey, data):
    # return base64.b64encode(str((privatekey.sign(data, ''))[0]).encode())
    return privatekey.sign(bytes(data))


def verifying(publickey, data, sign):
    # return publickey.verify(data, (int(base64.b64decode(sign)),))
    assert publickey.verify(sign, bytes(data))
    return True


def hash_block(block: grpc_pb2.Block) -> str:
    hash_str = block.SerializeToString()
    s = hashlib.sha256()  # Get the hash algorithm.
    s.update(hash_str)  # Hash the data.
    hash = s.hexdigest()  # Get he hash value.
    return hash


global pre_prepare_receive


class Blockchain:
    def __init__(self):
        self.node_id = p2p.PORT
        self.nodes = set()
        self.chain = []
        self.role = self.leader_select()
        self.lastBlock = None
        self.prepare_message_receive = []
        self.commit_message_receive = []
        # Genesis block
        block = self.create_block(None)
        self.add_block(block)

    def add_block(self, block):
        self.chain.append(block)
        self.lastBlock = block

    def create_block(self,tensor) -> grpc_pb2.Block:
        if self.lastBlock is None:
            block = grpc_pb2.Block(
                height=1,
                timestamp=int(time.time()),
                previoushash=b'',
                txshash=[],
                krumgrad=b''
            )
        else:
            block = grpc_pb2.Block(
                height=self.lastBlock.height + 1,
                timestamp=int(time.time()),
                previoushash=hash_block(self.lastBlock),
                txshash=[],
                krumgrad=tensor
            )
        return block

    def leader_select(self):
        if self.node_id == "50051":
            return 'leader'
        else:
            return 'member'

    def block_hash(self, Block):
        return Block.block_hash

    def block_height(self, block):
        return block.height

    def block_tx(self, block):
        return block.tx

    def consensus_process(self,tensor):
        # PRE-PREPARE
        global block,pre_prepare_receive
        if self.role == 'leader':
            block = self.create_block(tensor)
            t = threading.Thread(target=self.pre_prepare(block))
            t.start()
            t.join()
        # PREPARE
        if self.role=='member':
            for i in range(10):
                time.sleep(5)
                if pre_prepare_receive is not None:
                    t = threading.Thread(target=self.prepare(pre_prepare_receive,tensor))
                    t.start()
                    t.join()
                    count1 = 0
                    time.sleep(3)
                    for response in self.prepare_message_receive:
                        count1 += 1
                    if count1 > len(self.nodes) * 2 / 3:
                        # COMMIT
                        t = threading.Thread(target=self.commit())
                        t.start()
                        t.join()
                        count2 = 0
                        for commit_response in self.commit_message_receive:
                            count2 += 1
                        if count2 > len(self.nodes) * 2 / 3:
                            self.add_block(pre_prepare_receive)
                            pre_prepare_receive = None
                            break
                    pre_prepare_receive = None
                    break
                else:
                    continue
        else:
            time.sleep(5)
            t = threading.Thread(target=self.prepare(block,tensor))
            t.start()
            t.join()
            count1 = 0
            time.sleep(3)
            for response in self.prepare_message_receive:
                count1 += 1
            if count1 > len(self.nodes) * 2 / 3:
                # COMMIT
                t = threading.Thread(target=self.commit())
                t.start()
                t.join()
                count2 = 0
                for commit_response in self.commit_message_receive:
                    count2 += 1
                if count2 > len(self.nodes) * 2 / 3:
                    self.add_block(block)
    # PRE-PREPARE
    def pre_prepare(self, block):
        sk = SigningKey.from_string(bytes.fromhex(svkey[self.node_id][0]), curve=NIST384p)
        request = grpc_pb2.PrePrepareMsg()
        request.data.node_id = self.node_id
        # https://stackoverflow.com/questions/18376190/attributeerror-assignment-not-allowed-to-composite-field-task-in-protocol-mes/22771612#22771612
        request.data.block.CopyFrom(block)
        a = request.data.SerializeToString()
        request.signature = signing(sk, a)
        self_node = set()
        self_node.add(p2p.SELF_IP_PORT)
        print(self_node)
        nodes = set(p2p.Node.get_nodes_list()) - self_node
        print("print nodes in broadcast:")
        print(nodes)
        for i in nodes:
            channel = grpc.insecure_channel(i)
            stub = grpc_pb2_grpc.ConsensusStub(channel)
            try:
                # print("PRE-PREPARE checkpoint 1")
                response = stub.PrePrepare(request)
                # print("PRE-PREPARE checkpoint 2")
                print(response.Result)
            except:
                print("CONNECTION FAILED IN PRE—PREPARE PHASE!")
                # PREPARE_flag=False
                break


    # PREPARE PHASE
    # 1. verify pre_prepare message
    # 2. broadcast prepare message if step 1 is passed
    def prepare(self, block, tensor):
        sk = SigningKey.from_string(bytes.fromhex(svkey[self.node_id][0]), curve=NIST384p)
        request = grpc_pb2.PrepareMsg()
        request.data.node_id = self.node_id
        # krum_grad=Tensor.deserialize_torch_tensor(block.krumgrad)
        # krum_grad1=Tensor.deserialize_torch_tensor(tensor)
        if tensor == block.krumgrad:
        # if krum_grad.equal(krum_grad1):
            request.data.vote = '1'
        else:
            request.data.vote='0'
        a = request.data.SerializeToString()
        request.signature = signing(sk, a)
        self_node = set()
        self_node.add(p2p.SELF_IP_PORT)
        print(self_node)
        nodes = set(p2p.Node.get_nodes_list()) - self_node
        print("print PERPARE nodes in broadcast:")
        print(nodes)
        for i in nodes:
            channel = grpc.insecure_channel(i)
            stub = grpc_pb2_grpc.ConsensusStub(channel)
            try:
                # print("PREPARE checkpoint 1")
                response = stub.Prepare(request)
                # print("PREPARE checkpoint 2")
                print(response.Result)
                # PREPARE_flag=True
                self.prepare_message_receive.append(response)
            except:
                print("CONNECTION FAILED IN PREPARE PHASE!")
                # PREPARE_flag=False
                break

    # COMMIT PHASE
    # 1. verify if 2N/3 prepare messages are received
    # 2. broadcast commit message if step 1 is passed
    def commit(self):
        sk = SigningKey.from_string(bytes.fromhex(svkey[self.node_id][0]), curve=NIST384p)
        request = grpc_pb2.CommitMsg()
        request.data.node_id = self.node_id
        request.data.vote = '1'
        a = request.data.SerializeToString()
        request.signature = signing(sk, a)
        self_node = set()
        self_node.add(p2p.SELF_IP_PORT)
        print(self_node)
        nodes = set(p2p.Node.get_nodes_list()) - self_node
        print("print nodes in broadcast:")
        print(nodes)
        for i in nodes:
            channel = grpc.insecure_channel(i)
            stub = grpc_pb2_grpc.ConsensusStub(channel)
            try:
                # print("COMMIT checkpoint 1")
                response = stub.Commit(request)
                # print("COMMIT checkpoint 2")
                print(response.Result)
                self.commit_message_receive.append(response)
            except:
                print("CONNECTION FAILED IN COMMIT PHASE!")
                # PREPARE_flag=False
                break


# Consensus using gRPC
class Consensus(grpc_pb2_grpc.ConsensusServicer):
    def PrePrepare(self, request, context):
        global pre_prepare_receive
        print("Pre-prepare message is received")
        vk = VerifyingKey.from_string(bytes.fromhex(svkey[request.data.node_id][1]), curve=NIST384p)
        a = request.data.SerializeToString()
        if verifying(vk, a, request.signature):
            pre_prepare_receive = grpc_pb2.Block()
            pre_prepare_receive.CopyFrom(request.data.block)
            print("Pre-prepare block is received")
            return grpc_pb2.ConsensusRsp(Result='Pre-prepare Received Successfully')

    def Prepare(self, request, context):
        print("Prepare message is received")
        vk = VerifyingKey.from_string(bytes.fromhex(svkey[request.data.node_id][1]), curve=NIST384p)
        a = request.data.SerializeToString()
        if verifying(vk, a, request.signature):
            if request.data.vote == '1':
                print("vote=====1")
                return grpc_pb2.ConsensusRsp(Result='Prepare Received Successfully')

    def Commit(self, request, context):
        print("Commit message is received")
        vk = VerifyingKey.from_string(bytes.fromhex(svkey[request.data.node_id][1]), curve=NIST384p)
        a = request.data.SerializeToString()
        if verifying(vk, a, request.signature):
            if request.data.vote == '1':
                return grpc_pb2.ConsensusRsp(Result='Commit Received Successfully')
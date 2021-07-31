import grpc_pb2
import synchronization
import bc_enum
import time
import hashlib
import threading


class Transaction():
    Transactions = {}
    TransactionsPool = {}

    def create(self, body):
        pb2 = grpc_pb2.Transaction(unixtime=str(time.time()), body=body, txhash="")
        strpb2 = pb2.SerializeToString()
        txhash = hashlib.sha256(strpb2).hexdigest()
        pb2.txhash = txhash
        self.pb2 = pb2
        Transaction.TransactionsPool[txhash] = self
        return self

    def txload(self, pb2tx):
        self.pb2 = pb2tx
        Transaction.TransactionsPool[self.pb2.txhash] = self

    def Broadcast(self):
        import p2p
        p2p.Node.broadcast(
            bc_enum.Service.SERVICE * bc_enum.Network.SYNCHRONIZATION + bc_enum.Synchronization.TRANSACTIONFROM,
            self.pb2)

    @staticmethod
    def loadtxs(txshash):
        for txhash in txshash:
            if txhash in Transaction.TransactionsPool:
                Transaction.Transactions[txhash] = Transaction.TransactionsPool.pop(txhash)
            elif not txhash in Transaction.Transactions:
                Transaction.Transactions[txhash] = ""
            else:
                pass
                # print("It should not happen. transaction loadtx()")

    @staticmethod
    def getPoolList():
        result = []
        iterator = Transaction.TransactionsPool.keys()
        try:
            while True:
                txhash = iterator.next()
                result.append(txhash)
        except Exception as e:
            pass
        return result

    @staticmethod
    def getDictList():
        result = []
        iterator = Transaction.Transactions.keys()
        try:
            result.append(iterator.next())
        except Exception as e:
            pass
        return result

    @staticmethod
    def sync():
        import p2p
        while True:
            item = Transaction.Transactions.items()
            try:
                while True:
                    itemTuple = item.next()
                    if itemTuple[1] == "":
                        # print ("<= txhash:%s" % itemTuple[0])
                        msg = grpc_pb2.Message(value=itemTuple[0])
                        p2p.Node.broadcast(
                            bc_enum.Service.SERVICE * bc_enum.Network.SYNCHRONIZATION + bc_enum.Synchronization.TRANSACTIONFROM,
                            msg)
            except Exception as e:
                pass

    @staticmethod
    def TransactionFromRecv(pb2tx):
        if pb2tx.body == "":
            return
        print("=> unixtime:%s\tbody:%s" % (pb2tx.unixtime, pb2tx.body))
        tx = Transaction()
        tx.pb2 = pb2tx
        Transaction.Transactions[tx.pb2.txhash] = tx


threading.Thread(target=Transaction.sync).start()

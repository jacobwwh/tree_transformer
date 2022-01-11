
class dataiterator(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.end_of_data = False
        self.start_position = 0
        self.data=data
        self.end = len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.end_of_data:
            raise StopIteration
        ss = self.start_position
        ee = self.start_position + self.batch_size
        self.start_position += self.batch_size
        if ee >= self.end:
            self.end_of_data = True
            #ss = self.end - self.batch_size
            ee=self.end
        return self.data[ss:ee]

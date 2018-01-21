package text

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// ReadText from a file
func ReadText(s *op.Scope, fileName string) (indices tf.Output) {
	read := op.ReadFile(s, op.Const(s.SubScope("file_name"), fileName))
	indices = op.DecodeRaw(s, read, tf.Uint8)
	return
}

// OneHot embed chars
func OneHot(s *op.Scope, indices tf.Output) (embeded tf.Output) {
	embeded = op.OneHot(s,
		indices,
		op.Const(s.SubScope("depth"), int32(256)),
		op.Const(s.SubScope("on_value"), float32(1)),
		op.Const(s.SubScope("on_value"), float32(0)),
	)
	return
}

// ToCharByte converts one-hot encoding back to one byte
func ToCharByte(s *op.Scope, oneHot tf.Output) (charByte tf.Output) {
	charByte = op.Cast(s, op.ArgMax(s, oneHot, op.Const(s, int32(-1))), tf.Uint8)
	return
}

// Split the text into chunks
func Split(s *op.Scope, data tf.Output, seqLen, numSeq int64) (seqs tf.Output) {
	slice := op.Slice(s, data, op.Const(s.SubScope("begin"), []int64{0}), op.Const(s.SubScope("size"), []int64{numSeq * seqLen}))
	seqs = op.Reshape(s, slice, op.Const(s.SubScope("shape"), []int64{numSeq, seqLen}))
	return
}

// Offset returns two version of input which are one element in the outer dimensions.
func Offset(s *op.Scope, input tf.Output) (x, y tf.Output) {
	nilChar := op.Const(s, []uint8{0})
	xs := s.SubScope("x")
	x = op.Concat(xs, op.Const(xs, int32(0)), []tf.Output{nilChar, input})
	ys := s.SubScope("y")
	y = op.Concat(ys, op.Const(ys, int32(0)), []tf.Output{input, nilChar})
	return
}

// NextSeqBatch returns a pair of the curent char and the next char
func NextSeqBatch(s *op.Scope, fileName string, seqLen, batchSize, numSeqs int64, seed int64) (curentChar, nextChar tf.Output, init *tf.Operation) {
	chars := ReadText(s.SubScope("read"), fileName)
	x, y := Offset(s.SubScope("offset"), chars)
	xs := s.SubScope("x")
	xSplit := OneHot(xs, Split(xs, x, seqLen, numSeqs))
	ys := s.SubScope("y")
	ySplit := OneHot(ys, Split(ys, y, seqLen, numSeqs))
	seedOutput := op.Const(s, seed)
	outputTypes := []tf.DataType{tf.Float, tf.Float}
	preBatchOutputShapes := []tf.Shape{tf.MakeShape(seqLen, 256), tf.MakeShape(seqLen, 256)}
	outputShapes := []tf.Shape{tf.MakeShape(batchSize, seqLen, 256), tf.MakeShape(batchSize, seqLen, 256)}
	dataset := op.TensorSliceDataset(s, []tf.Output{xSplit, ySplit}, outputShapes)
	repeatDataset := op.RepeatDataset(s, dataset, op.Const(s.SubScope("count"), int64(-1)), outputTypes, preBatchOutputShapes)
	shuffleDataset := op.ShuffleDataset(s,
		repeatDataset,
		op.Const(s.SubScope("buffer_size"), int64(100)),
		seedOutput,
		seedOutput,
		outputTypes,
		preBatchOutputShapes,
	)
	batchDataset := op.BatchDataset(s, shuffleDataset, op.Const(s.SubScope("batch_size"), batchSize), outputTypes, outputShapes)
	iterator := op.Iterator(s, "", "", outputTypes, outputShapes)
	next := op.IteratorGetNext(s, iterator, outputTypes, outputShapes)
	init = op.MakeIterator(s, batchDataset, iterator)
	curentChar = next[0]
	nextChar = next[1]
	return
}

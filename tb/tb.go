package tb

import (
	"bytes"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// WriteGraphSummary takes a graph and writes the summary to the given logDir
func WriteGraphSummary(g *tf.Graph, logDir string) (err error) {
	w := bytes.Buffer{}
	_, err = g.WriteTo(&w)
	if err != nil {
		return
	}
	s := op.NewScope()
	writer := op.SummaryWriter(s)
	createSummaryWriter := op.CreateSummaryFileWriter(s,
		writer,
		op.Const(s.SubScope("log_dir"), logDir),
		op.Const(s.SubScope("max_queue"), int32(1)),
		op.Const(s.SubScope("flush_millis"), int32(1000)),
		op.Const(s.SubScope("filename_suffix"), "graph"),
	)
	closeSummaryWriter := op.CloseSummaryWriter(s, writer)
	write := op.WriteGraphSummary(s, writer, op.Const(s.SubScope("step"), int64(0)), op.Const(s.SubScope("tensor"), string(w.Bytes())))
	graph, err := s.Finalize()
	if err != nil {
		return
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{createSummaryWriter})
	if err != nil {
		return
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{write})
	if err != nil {
		return
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{closeSummaryWriter})
	if err != nil {
		return
	}
	return
}

// LogOP can be called to create a SummaryWriter for some part of a graph.
type LogOP struct {
	Name   string
	OPfunc func(s *op.Scope, writer tf.Output, tag tf.Output, step tf.Output) (writerOP *tf.Operation)
}

// MakeHistLogOP creates a LogOP struct for a HistogramSummary TensorBoard writer.
func MakeHistLogOP(values tf.Output, name string) LogOP {
	return LogOP{
		Name: name,
		OPfunc: func(s *op.Scope, writer tf.Output, tag tf.Output, step tf.Output) (writerOP *tf.Operation) {
			return op.WriteHistogramSummary(s, writer, step, tag, values)
		},
	}
}

func touint8(s *op.Scope, floatVal tf.Output) tf.Output {
	uint8Max := op.Const(s, float32(255))
	return op.Cast(s, op.Minimum(s, op.Mul(s, floatVal, uint8Max), uint8Max), tf.Uint8)
}

// MakePlusMinusOneImageLogOP creates a LogOP struct for a ImageSummary TensorBoard writer.
// It assumes that the input is a float 2d tensor between -1 and 1.
func MakePlusMinusOneImageLogOP(data tf.Output, name string) LogOP {
	return LogOP{
		Name: name,
		OPfunc: func(s *op.Scope, writer tf.Output, tag tf.Output, step tf.Output) (writeImage *tf.Operation) {
			zero := op.Const(s.SubScope("zero"), float32(0))
			pos := touint8(s.SubScope("pos"), op.Maximum(s, zero, data))
			neg := touint8(s.SubScope("neg"), op.Abs(s, op.Minimum(s, zero, data)))
			zeros := op.Fill(s, op.Shape(s, pos), op.Const(s.SubScope("uint80"), uint8(0)))
			image := op.Pack(s, []tf.Output{neg, pos, zeros}, op.PackAxis(2))
			images := op.ExpandDims(s, image, op.Const(s.SubScope("expand_dim"), int32(0)))
			writeImage = op.WriteImageSummary(s, writer, step, tag, images, op.Const(s.SubScope("bad_color"), []uint8{0, 0, 0}))
			return
		},
	}
}

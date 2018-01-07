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

package text

import (
	"fmt"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func TestNextSeqBatch(t *testing.T) {
	s := op.NewScope()
	x, y, init := NextSeqBatch(s, "jabberwock.txt", 50, 3, 4, 42)
	xid := ToCharByte(s.SubScope("x"), x)
	yid := ToCharByte(s.SubScope("y"), y)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{init})
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{xid, yid}, nil)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(results[0].Shape())
	fmt.Println(string([]byte([]uint8(results[0].Value().([][]byte)[0]))))
}

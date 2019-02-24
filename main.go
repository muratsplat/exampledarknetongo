package main

// #cgo CFLAGS: -I libdarknet.so
// #cgo LDFLAGS: -L. -ldarknet
// #include <stdlib.h>
// #include "darknet.h"
import "C"
import "fmt"

var (
	netDark      *C.network
	metaDataDark C.metadata
)

func init() {

}

func main() {
	cfgPath := C.CString("cfg/yolov3-tiny.cfg")
	weightsPath := C.CString("weights/tiny.weights")
	netDark = C.load_network(cfgPath, weightsPath, 0)
	metaDataDark = C.get_metadata(C.CString("data/coco.names"))
	img := C.load_image_color(C.CString("data/dog.jpg"), 0, 0)
	nboxes := C.int(0)
	nms := C.float(0.45)
	C.network_predict_image(netDark, img)
	dets := C.get_network_boxes(netDark, img.w, img.h, C.float(0.5), C.float(0.5), nil, 0, &nboxes)
	if nms > 0 {
		C.do_nms_obj(dets, nboxes, metaDataDark.classes, nms)
	}

	fmt.Println(dets)

}

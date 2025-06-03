#!/bin/bash
echo "[ascend310b] Generating Pows_076e20dfb5ef675fa9e9556c39bee126 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=pows --input_param=/root/ascend_s4/Pows/build_out/op_kernel/binary/ascend310b/gen/Pows_076e20dfb5ef675fa9e9556c39bee126_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Pows_076e20dfb5ef675fa9e9556c39bee126.json ; then
  echo "$2/Pows_076e20dfb5ef675fa9e9556c39bee126.json not generated!"
  exit 1
fi

if ! test -f $2/Pows_076e20dfb5ef675fa9e9556c39bee126.o ; then
  echo "$2/Pows_076e20dfb5ef675fa9e9556c39bee126.o not generated!"
  exit 1
fi
echo "[ascend310b] Generating Pows_076e20dfb5ef675fa9e9556c39bee126 Done"

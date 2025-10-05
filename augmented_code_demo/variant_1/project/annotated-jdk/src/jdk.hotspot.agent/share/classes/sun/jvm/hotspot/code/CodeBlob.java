/*
    @Positive
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package sun.jvm.hotspot.code;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import sun.jvm.hotspot.compiler.ImmutableOopMap;
    @Positive
import sun.jvm.hotspot.compiler.ImmutableOopMapSet;
    @Positive
import sun.jvm.hotspot.debugger.Address;
    @Positive
import sun.jvm.hotspot.runtime.VM;
    @Positive
import sun.jvm.hotspot.runtime.VMObject;
    @Positive
import sun.jvm.hotspot.types.AddressField;
    @Positive
import sun.jvm.hotspot.types.CIntegerField;
    @Positive
import sun.jvm.hotspot.types.Type;
    @Positive
import sun.jvm.hotspot.types.TypeDataBase;
    @Positive
import sun.jvm.hotspot.utilities.Assert;
    @Positive
import sun.jvm.hotspot.utilities.CStringUtilities;
    @Positive
import java.io.PrintStream;
    @Positive
import sun.jvm.hotspot.utilities.Observable;
    @Positive
import sun.jvm.hotspot.utilities.Observer;

    @Positive
public class CodeBlob extends VMObject {

    @Positive
    public CodeBlob(Address addr) {
    @Positive
    }

    @Positive
    protected static int matcherInterpreterFramePointerReg;

    @Positive
    public Address headerBegin();

    @Positive
    public Address headerEnd();

    @Positive
    public Address contentBegin();

    @Positive
    public Address contentEnd();

    @Positive
    public Address codeBegin();

    @Positive
    public Address codeEnd();

    @Positive
    public Address dataBegin();

    @Positive
    public Address dataEnd();

    @Positive
    public long getFrameCompleteOffset();

    @Positive
    public int getDataOffset();

    @Positive
    public int getSize();

    @Positive
    public int getHeaderSize();

    @Positive
    public long getFrameSizeWords();

    @Positive
    public String getName();

    @Positive
    public ImmutableOopMapSet getOopMaps();

    @Positive
    public boolean isBufferBlob();

    @Positive
    public boolean isCompiled();

    @Positive
    public boolean isNMethod();

    @Positive
    public boolean isRuntimeStub();

    @Positive
    public boolean isDeoptimizationStub();

    @Positive
    public boolean isUncommonTrapStub();

    @Positive
    public boolean isExceptionStub();

    @Positive
    public boolean isSafepointStub();

    @Positive
    public boolean isAdapterBlob();

    @Positive
    public boolean isJavaMethod();

    @Positive
    public boolean isNativeMethod();

    @Positive
    public boolean isOSRMethod();

    @Positive
    public NMethod asNMethodOrNull();

    @Positive
    public int getContentSize();

    @Positive
    public int getCodeSize();

    @Positive
    public int getDataSize();

    @Positive
    public boolean blobContains(Address addr);

    @Positive
    public boolean contentContains(Address addr);

    @Positive
    public boolean codeContains(Address addr);

    @Positive
    public boolean dataContains(Address addr);

    @Positive
    @Pure
    @Positive
    public boolean contains(Address addr);

    @Positive
    public boolean isFrameCompleteAt(Address a);

    @Positive
    public boolean isZombie();

    @Positive
    public boolean isLockedByVM();

    @Positive
    public ImmutableOopMap getOopMapForReturnAddress(Address returnAddress, boolean debugging);

    @Positive
    public long getFrameSize();

    @Positive
    public boolean callerMustGCArguments();

    @Positive
    public void print();

    @Positive
    public void printOn(PrintStream tty);

    @Positive
    protected void printComponentsOn(PrintStream tty);
    @Positive
}

// CFWR semantic augmentation - variant 1

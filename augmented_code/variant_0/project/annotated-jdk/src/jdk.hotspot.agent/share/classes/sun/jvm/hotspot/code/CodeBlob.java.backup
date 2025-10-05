/*
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package sun.jvm.hotspot.code;

import org.checkerframework.dataflow.qual.Pure;
import sun.jvm.hotspot.compiler.ImmutableOopMap;
import sun.jvm.hotspot.compiler.ImmutableOopMapSet;
import sun.jvm.hotspot.debugger.Address;
import sun.jvm.hotspot.runtime.VM;
import sun.jvm.hotspot.runtime.VMObject;
import sun.jvm.hotspot.types.AddressField;
import sun.jvm.hotspot.types.CIntegerField;
import sun.jvm.hotspot.types.Type;
import sun.jvm.hotspot.types.TypeDataBase;
import sun.jvm.hotspot.utilities.Assert;
import sun.jvm.hotspot.utilities.CStringUtilities;
import java.io.PrintStream;
import sun.jvm.hotspot.utilities.Observable;
import sun.jvm.hotspot.utilities.Observer;

public class CodeBlob extends VMObject {

    public CodeBlob(Address addr) {
    }

    protected static int matcherInterpreterFramePointerReg;

    public Address headerBegin();

    public Address headerEnd();

    public Address contentBegin();

    public Address contentEnd();

    public Address codeBegin();

    public Address codeEnd();

    public Address dataBegin();

    public Address dataEnd();

    public long getFrameCompleteOffset();

    public int getDataOffset();

    public int getSize();

    public int getHeaderSize();

    public long getFrameSizeWords();

    public String getName();

    public ImmutableOopMapSet getOopMaps();

    public boolean isBufferBlob();

    public boolean isCompiled();

    public boolean isNMethod();

    public boolean isRuntimeStub();

    public boolean isDeoptimizationStub();

    public boolean isUncommonTrapStub();

    public boolean isExceptionStub();

    public boolean isSafepointStub();

    public boolean isAdapterBlob();

    public boolean isJavaMethod();

    public boolean isNativeMethod();

    public boolean isOSRMethod();

    public NMethod asNMethodOrNull();

    public int getContentSize();

    public int getCodeSize();

    public int getDataSize();

    public boolean blobContains(Address addr);

    public boolean contentContains(Address addr);

    public boolean codeContains(Address addr);

    public boolean dataContains(Address addr);

    @Pure
    public boolean contains(Address addr);

    public boolean isFrameCompleteAt(Address a);

    public boolean isZombie();

    public boolean isLockedByVM();

    public ImmutableOopMap getOopMapForReturnAddress(Address returnAddress, boolean debugging);

    public long getFrameSize();

    public boolean callerMustGCArguments();

    public void print();

    public void printOn(PrintStream tty);

    protected void printComponentsOn(PrintStream tty);
}

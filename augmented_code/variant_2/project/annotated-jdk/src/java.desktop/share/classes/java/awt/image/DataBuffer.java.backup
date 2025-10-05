/*
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package java.awt.image;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import sun.java2d.StateTrackable.State;
import static sun.java2d.StateTrackable.State.*;
import sun.java2d.StateTrackableDelegate;
import sun.awt.image.SunWritableRaster;
import java.lang.annotation.Native;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class DataBuffer {

    @Native
    public static final int TYPE_BYTE;

    @Native
    public static final int TYPE_USHORT;

    @Native
    public static final int TYPE_SHORT;

    @Native
    public static final int TYPE_INT;

    @Native
    public static final int TYPE_FLOAT;

    @Native
    public static final int TYPE_DOUBLE;

    @Native
    public static final int TYPE_UNDEFINED;

    protected int dataType;

    protected int banks;

    protected int offset;

    protected int size;

    protected int[] offsets;

    public static int getDataTypeSize(int type);

    protected DataBuffer(int dataType, int size) {
    }

    protected DataBuffer(int dataType, int size, int numBanks) {
    }

    protected DataBuffer(int dataType, int size, int numBanks, int offset) {
    }

    protected DataBuffer(int dataType, int size, int numBanks, int[] offsets) {
    }

    public int getDataType();

    public int getSize();

    public int getOffset();

    public int[] getOffsets();

    public int getNumBanks();

    public int getElem(int i);

    public abstract int getElem(int bank, int i);

    public void setElem(int i, int val);

    public abstract void setElem(int bank, int i, int val);

    public float getElemFloat(int i);

    public float getElemFloat(int bank, int i);

    public void setElemFloat(int i, float val);

    public void setElemFloat(int bank, int i, float val);

    public double getElemDouble(int i);

    public double getElemDouble(int bank, int i);

    public void setElemDouble(int i, double val);

    public void setElemDouble(int bank, int i, double val);

    static int[] toIntArray(Object obj);
}

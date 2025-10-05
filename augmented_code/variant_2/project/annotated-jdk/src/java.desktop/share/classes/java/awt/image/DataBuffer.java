/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
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
package java.awt.image;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.java2d.StateTrackable.State;
    @Positive
import static sun.java2d.StateTrackable.State.*;
    @Positive
import sun.java2d.StateTrackableDelegate;
    @Positive
import sun.awt.image.SunWritableRaster;
    @Positive
import java.lang.annotation.Native;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class DataBuffer {

    @Positive
    @Native
    @Positive
    public static final int TYPE_BYTE;

    @Positive
    @Native
    @Positive
    public static final int TYPE_USHORT;

    @Positive
    @Native
    @Positive
    public static final int TYPE_SHORT;

    @Positive
    @Native
    @Positive
    public static final int TYPE_INT;

    @Positive
    @Native
    @Positive
    public static final int TYPE_FLOAT;

    @Positive
    @Native
    @Positive
    public static final int TYPE_DOUBLE;

    @Positive
    @Native
    @Positive
    public static final int TYPE_UNDEFINED;

    @Positive
    protected int dataType;

    @Positive
    protected int banks;

    @Positive
    protected int offset;

    @Positive
    protected int size;

    @Positive
    protected int[] offsets;

    @Positive
    public static int getDataTypeSize(int type);

    @Positive
    protected DataBuffer(int dataType, int size) {
    @Positive
    }

    @Positive
    protected DataBuffer(int dataType, int size, int numBanks) {
    @Positive
    }

    @Positive
    protected DataBuffer(int dataType, int size, int numBanks, int offset) {
    @Positive
    }

    @Positive
    protected DataBuffer(int dataType, int size, int numBanks, int[] offsets) {
    @Positive
    }

    @Positive
    public int getDataType();

    @Positive
    public int getSize();

    @Positive
    public int getOffset();

    @Positive
    public int[] getOffsets();

    @Positive
    public int getNumBanks();

    @Positive
    public int getElem(int i);

    @Positive
    public abstract int getElem(int bank, int i);

    @Positive
    public void setElem(int i, int val);

    @Positive
    public abstract void setElem(int bank, int i, int val);

    @Positive
    public float getElemFloat(int i);

    @Positive
    public float getElemFloat(int bank, int i);

    @Positive
    public void setElemFloat(int i, float val);

    @Positive
    public void setElemFloat(int bank, int i, float val);

    @Positive
    public double getElemDouble(int i);

    @Positive
    public double getElemDouble(int bank, int i);

    @Positive
    public void setElemDouble(int i, double val);

    @Positive
    public void setElemDouble(int bank, int i, double val);

    @Positive
    static int[] toIntArray(Object obj);
    @Positive
}

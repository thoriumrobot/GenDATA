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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.Arrays;

    @Positive
public class SinglePixelPackedSampleModel extends SampleModel {

    @Positive
    public SinglePixelPackedSampleModel(int dataType, int w, int h, int[] bitMasks) {
    @Positive
    }

    @Positive
    public SinglePixelPackedSampleModel(int dataType, int w, int h, int scanlineStride, int[] bitMasks) {
    @Positive
    }

    @Positive
    public int getNumDataElements();

    @Positive
    public SampleModel createCompatibleSampleModel(int w, int h);

    @Positive
    public DataBuffer createDataBuffer();

    @Positive
    public int[] getSampleSize();

    @Positive
    public int getSampleSize(int band);

    @Positive
    public int getOffset(int x, int y);

    @Positive
    public int[] getBitOffsets();

    @Positive
    public int[] getBitMasks();

    @Positive
    public int getScanlineStride();

    @Positive
    public SampleModel createSubsetSampleModel(int[] bands);

    @Positive
    public Object getDataElements(int x, int y, Object obj, DataBuffer data);

    @Positive
    public int[] getPixel(int x, int y, int[] iArray, DataBuffer data);

    @Positive
    public int[] getPixels(int x, int y, int w, int h, int[] iArray, DataBuffer data);

    @Positive
    public int getSample(int x, int y, int b, DataBuffer data);

    @Positive
    public int[] getSamples(int x, int y, int w, int h, int b, int[] iArray, DataBuffer data);

    @Positive
    public void setDataElements(int x, int y, Object obj, DataBuffer data);

    @Positive
    public void setPixel(int x, int y, int[] iArray, DataBuffer data);

    @Positive
    public void setPixels(int x, int y, int w, int h, int[] iArray, DataBuffer data);

    @Positive
    public void setSample(int x, int y, int b, int s, DataBuffer data);

    @Positive
    public void setSamples(int x, int y, int w, int h, int b, int[] iArray, DataBuffer data);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0

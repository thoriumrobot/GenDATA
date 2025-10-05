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

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Arrays;

public class SinglePixelPackedSampleModel extends SampleModel {

    public SinglePixelPackedSampleModel(int dataType, int w, int h, int[] bitMasks) {
    }

    public SinglePixelPackedSampleModel(int dataType, int w, int h, int scanlineStride, int[] bitMasks) {
    }

    public int getNumDataElements();

    public SampleModel createCompatibleSampleModel(int w, int h);

    public DataBuffer createDataBuffer();

    public int[] getSampleSize();

    public int getSampleSize(int band);

    public int getOffset(int x, int y);

    public int[] getBitOffsets();

    public int[] getBitMasks();

    public int getScanlineStride();

    public SampleModel createSubsetSampleModel(int[] bands);

    public Object getDataElements(int x, int y, Object obj, DataBuffer data);

    public int[] getPixel(int x, int y, int[] iArray, DataBuffer data);

    public int[] getPixels(int x, int y, int w, int h, int[] iArray, DataBuffer data);

    public int getSample(int x, int y, int b, DataBuffer data);

    public int[] getSamples(int x, int y, int w, int h, int b, int[] iArray, DataBuffer data);

    public void setDataElements(int x, int y, Object obj, DataBuffer data);

    public void setPixel(int x, int y, int[] iArray, DataBuffer data);

    public void setPixels(int x, int y, int w, int h, int[] iArray, DataBuffer data);

    public void setSample(int x, int y, int b, int s, DataBuffer data);

    public void setSamples(int x, int y, int w, int h, int b, int[] iArray, DataBuffer data);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();
}

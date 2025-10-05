/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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

public class ComponentSampleModel extends SampleModel {

    protected int[] bandOffsets;

    protected int[] bankIndices;

    protected int numBands;

    protected int numBanks;

    protected int scanlineStride;

    protected int pixelStride;

    public ComponentSampleModel(int dataType, int w, int h, int pixelStride, int scanlineStride, int[] bandOffsets) {
    }

    public ComponentSampleModel(int dataType, int w, int h, int pixelStride, int scanlineStride, int[] bankIndices, int[] bandOffsets) {
    }

    int[] orderBands(int[] orig, int step);

    public SampleModel createCompatibleSampleModel(int w, int h);

    public SampleModel createSubsetSampleModel(int[] bands);

    public DataBuffer createDataBuffer();

    public int getOffset(int x, int y);

    public int getOffset(int x, int y, int b);

    public final int[] getSampleSize();

    public final int getSampleSize(int band);

    public final int[] getBankIndices();

    public final int[] getBandOffsets();

    public final int getScanlineStride();

    public final int getPixelStride();

    public final int getNumDataElements();

    public Object getDataElements(int x, int y, Object obj, DataBuffer data);

    public int[] getPixel(int x, int y, int[] iArray, DataBuffer data);

    public int[] getPixels(int x, int y, int w, int h, int[] iArray, DataBuffer data);

    public int getSample(int x, int y, int b, DataBuffer data);

    public float getSampleFloat(int x, int y, int b, DataBuffer data);

    public double getSampleDouble(int x, int y, int b, DataBuffer data);

    public int[] getSamples(int x, int y, int w, int h, int b, int[] iArray, DataBuffer data);

    public void setDataElements(int x, int y, Object obj, DataBuffer data);

    public void setPixel(int x, int y, int[] iArray, DataBuffer data);

    public void setPixels(int x, int y, int w, int h, int[] iArray, DataBuffer data);

    public void setSample(int x, int y, int b, int s, DataBuffer data);

    public void setSample(int x, int y, int b, float s, DataBuffer data);

    public void setSample(int x, int y, int b, double s, DataBuffer data);

    public void setSamples(int x, int y, int w, int h, int b, int[] iArray, DataBuffer data);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();
}

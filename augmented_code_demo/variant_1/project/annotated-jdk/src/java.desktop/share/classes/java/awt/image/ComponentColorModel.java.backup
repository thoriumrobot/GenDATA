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
import java.awt.color.ColorSpace;
import java.awt.color.ICC_ColorSpace;
import java.util.Arrays;

public class ComponentColorModel extends ColorModel {

    public ComponentColorModel(ColorSpace colorSpace, int[] bits, boolean hasAlpha, boolean isAlphaPremultiplied, int transparency, int transferType) {
    }

    public ComponentColorModel(ColorSpace colorSpace, boolean hasAlpha, boolean isAlphaPremultiplied, int transparency, int transferType) {
    }

    public int getRed(int pixel);

    public int getGreen(int pixel);

    public int getBlue(int pixel);

    public int getAlpha(int pixel);

    public int getRGB(int pixel);

    public int getRed(Object inData);

    public int getGreen(Object inData);

    public int getBlue(Object inData);

    public int getAlpha(Object inData);

    public int getRGB(Object inData);

    public Object getDataElements(int rgb, Object pixel);

    public int[] getComponents(int pixel, int[] components, int offset);

    public int[] getComponents(Object pixel, int[] components, int offset);

    public int[] getUnnormalizedComponents(float[] normComponents, int normOffset, int[] components, int offset);

    public float[] getNormalizedComponents(int[] components, int offset, float[] normComponents, int normOffset);

    public int getDataElement(int[] components, int offset);

    public Object getDataElements(int[] components, int offset, Object obj);

    public int getDataElement(float[] normComponents, int normOffset);

    public Object getDataElements(float[] normComponents, int normOffset, Object obj);

    public float[] getNormalizedComponents(Object pixel, float[] normComponents, int normOffset);

    public ColorModel coerceData(WritableRaster raster, boolean isAlphaPremultiplied);

    public boolean isCompatibleRaster(Raster raster);

    public WritableRaster createCompatibleWritableRaster(int w, int h);

    public SampleModel createCompatibleSampleModel(int w, int h);

    public boolean isCompatibleSampleModel(SampleModel sm);

    public WritableRaster getAlphaRaster(WritableRaster raster);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();
}

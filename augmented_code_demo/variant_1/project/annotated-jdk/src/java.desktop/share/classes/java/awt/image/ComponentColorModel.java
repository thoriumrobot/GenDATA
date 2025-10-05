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
import java.awt.color.ColorSpace;
    @Positive
import java.awt.color.ICC_ColorSpace;
    @Positive
import java.util.Arrays;

    @Positive
public class ComponentColorModel extends ColorModel {

    @Positive
    public ComponentColorModel(ColorSpace colorSpace, int[] bits, boolean hasAlpha, boolean isAlphaPremultiplied, int transparency, int transferType) {
    @Positive
    }

    @Positive
    public ComponentColorModel(ColorSpace colorSpace, boolean hasAlpha, boolean isAlphaPremultiplied, int transparency, int transferType) {
    @Positive
    }

    @Positive
    public int getRed(int pixel);

    @Positive
    public int getGreen(int pixel);

    @Positive
    public int getBlue(int pixel);

    @Positive
    public int getAlpha(int pixel);

    @Positive
    public int getRGB(int pixel);

    @Positive
    public int getRed(Object inData);

    @Positive
    public int getGreen(Object inData);

    @Positive
    public int getBlue(Object inData);

    @Positive
    public int getAlpha(Object inData);

    @Positive
    public int getRGB(Object inData);

    @Positive
    public Object getDataElements(int rgb, Object pixel);

    @Positive
    public int[] getComponents(int pixel, int[] components, int offset);

    @Positive
    public int[] getComponents(Object pixel, int[] components, int offset);

    @Positive
    public int[] getUnnormalizedComponents(float[] normComponents, int normOffset, int[] components, int offset);

    @Positive
    public float[] getNormalizedComponents(int[] components, int offset, float[] normComponents, int normOffset);

    @Positive
    public int getDataElement(int[] components, int offset);

    @Positive
    public Object getDataElements(int[] components, int offset, Object obj);

    @Positive
    public int getDataElement(float[] normComponents, int normOffset);

    @Positive
    public Object getDataElements(float[] normComponents, int normOffset, Object obj);

    @Positive
    public float[] getNormalizedComponents(Object pixel, float[] normComponents, int normOffset);

    @Positive
    public ColorModel coerceData(WritableRaster raster, boolean isAlphaPremultiplied);

    @Positive
    public boolean isCompatibleRaster(Raster raster);

    @Positive
    public WritableRaster createCompatibleWritableRaster(int w, int h);

    @Positive
    public SampleModel createCompatibleSampleModel(int w, int h);

    @Positive
    public boolean isCompatibleSampleModel(SampleModel sm);

    @Positive
    public WritableRaster getAlphaRaster(WritableRaster raster);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 1

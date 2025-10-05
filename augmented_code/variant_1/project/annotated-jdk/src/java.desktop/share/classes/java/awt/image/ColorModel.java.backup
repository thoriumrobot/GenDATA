/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.index.qual.SameLen;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.color.ICC_ColorSpace;
import sun.java2d.cmm.CMSManager;
import sun.java2d.cmm.ColorTransform;
import sun.java2d.cmm.PCMM;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.Arrays;

@AnnotatedFor({ "index" })
public abstract class ColorModel implements Transparency {

    protected int pixel_bits;

    protected int transferType;

    @SuppressWarnings("removal")
    static void loadLibraries();

    public static ColorModel getRGBdefault();

    public ColorModel(@Positive int bits) {
    }

    protected ColorModel(@Positive int pixel_bits, @NonNegative int[] bits, ColorSpace cspace, boolean hasAlpha, boolean isAlphaPremultiplied, int transparency, int transferType) {
    }

    public final boolean hasAlpha();

    public final boolean isAlphaPremultiplied();

    public final int getTransferType();

    @NonNegative
    public int getPixelSize();

    @NonNegative
    public int getComponentSize(@IndexFor({ "this" }) int componentIdx);

    @NonNegative
    public int[] getComponentSize();

    public int getTransparency();

    @LengthOf({ "this" })
    public int getNumComponents();

    @IndexOrHigh({ "this" })
    public int getNumColorComponents();

    public abstract int getRed(int pixel);

    public abstract int getGreen(int pixel);

    public abstract int getBlue(int pixel);

    public abstract int getAlpha(int pixel);

    public int getRGB(int pixel);

    public int getRed(Object inData);

    public int getGreen(Object inData);

    public int getBlue(Object inData);

    public int getAlpha(Object inData);

    public int getRGB(Object inData);

    public Object getDataElements(int rgb, Object pixel);

    @NonNegative
    public int @SameLen({ "#2" }) @PolyValue [] getComponents(int pixel, int @PolyValue [] components, @IndexFor({ "#2" }) int offset);

    @NonNegative
    public int @SameLen({ "#2" }) @PolyValue [] getComponents(Object pixel, int @PolyValue [] components, @IndexFor({ "#2" }) int offset);

    public int @SameLen({ "#1", "#3" }) [] getUnnormalizedComponents(float @SameLen({ "#1", "#3" }) [] normComponents, @IndexFor({ "#1" }) int normOffset, int @SameLen({ "#1", "#3" }) [] components, @IndexFor({ "#3" }) int offset);

    public float @SameLen({ "#1", "#3" }) [] getNormalizedComponents(int @SameLen({ "#1", "#3" }) [] components, @IndexFor({ "#1" }) int offset, float @SameLen({ "#1", "#3" }) [] normComponents, @IndexFor({ "#3" }) int normOffset);

    public int getDataElement(int[] components, @IndexFor({ "#1" }) int offset);

    public Object getDataElements(int[] components, @IndexFor({ "#1" }) int offset, Object obj);

    public int getDataElement(float[] normComponents, @IndexFor({ "#1" }) int normOffset);

    public Object getDataElements(float[] normComponents, @IndexFor({ "#1" }) int normOffset, Object obj);

    public float[] getNormalizedComponents(Object pixel, float[] normComponents, @IndexFor({ "#2" }) int normOffset);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    public final ColorSpace getColorSpace();

    public ColorModel coerceData(WritableRaster raster, boolean isAlphaPremultiplied);

    public boolean isCompatibleRaster(Raster raster);

    public WritableRaster createCompatibleWritableRaster(int w, int h);

    public SampleModel createCompatibleSampleModel(int w, int h);

    public boolean isCompatibleSampleModel(SampleModel sm);

    @Deprecated()
    @SuppressWarnings("removal")
    public void finalize();

    public WritableRaster getAlphaRaster(WritableRaster raster);

    public String toString();

    static int getDefaultTransferType(int pixel_bits);

    static boolean isLinearRGBspace(ColorSpace cs);

    static boolean isLinearGRAYspace(ColorSpace cs);

    static byte[] getLinearRGB8TosRGB8LUT();

    static byte[] getsRGB8ToLinearRGB8LUT();

    static byte[] getLinearRGB16TosRGB8LUT();

    static short[] getsRGB8ToLinearRGB16LUT();

    static byte[] getGray8TosRGB8LUT(ICC_ColorSpace grayCS);

    static byte[] getLinearGray16ToOtherGray8LUT(ICC_ColorSpace grayCS);

    static byte[] getGray16TosRGB8LUT(ICC_ColorSpace grayCS);

    static short[] getLinearGray16ToOtherGray16LUT(ICC_ColorSpace grayCS);
}

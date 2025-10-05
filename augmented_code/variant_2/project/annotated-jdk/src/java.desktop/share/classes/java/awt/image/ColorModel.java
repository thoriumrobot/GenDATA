/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Transparency;
    @Positive
import java.awt.color.ColorSpace;
    @Positive
import java.awt.color.ICC_ColorSpace;
    @Positive
import sun.java2d.cmm.CMSManager;
    @Positive
import sun.java2d.cmm.ColorTransform;
    @Positive
import sun.java2d.cmm.PCMM;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Map;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.Arrays;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public abstract class ColorModel implements Transparency {

    @Positive
    protected int pixel_bits;

    @Positive
    protected int transferType;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static void loadLibraries();

    @Positive
    public static ColorModel getRGBdefault();

    @Positive
    public ColorModel(@Positive int bits) {
    @Positive
    }

    @Positive
    protected ColorModel(@Positive int pixel_bits, @NonNegative int[] bits, ColorSpace cspace, boolean hasAlpha, boolean isAlphaPremultiplied, int transparency, int transferType) {
    @Positive
    }

    @Positive
    public final boolean hasAlpha();

    @Positive
    public final boolean isAlphaPremultiplied();

    @Positive
    public final int getTransferType();

    @Positive
    @NonNegative
    @Positive
    public int getPixelSize();

    @Positive
    @NonNegative
    @Positive
    public int getComponentSize(@IndexFor({ "this" }) int componentIdx);

    @Positive
    @NonNegative
    @Positive
    public int[] getComponentSize();

    @Positive
    public int getTransparency();

    @Positive
    @LengthOf({ "this" })
    @Positive
    public int getNumComponents();

    @Positive
    @IndexOrHigh({ "this" })
    @Positive
    public int getNumColorComponents();

    @Positive
    public abstract int getRed(int pixel);

    @Positive
    public abstract int getGreen(int pixel);

    @Positive
    public abstract int getBlue(int pixel);

    @Positive
    public abstract int getAlpha(int pixel);

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
    @NonNegative
    @Positive
    public int @SameLen({ "#2" }) @PolyValue [] getComponents(int pixel, int @PolyValue [] components, @IndexFor({ "#2" }) int offset);

    @Positive
    @NonNegative
    @Positive
    public int @SameLen({ "#2" }) @PolyValue [] getComponents(Object pixel, int @PolyValue [] components, @IndexFor({ "#2" }) int offset);

    @Positive
    public int @SameLen({ "#1", "#3" }) [] getUnnormalizedComponents(float @SameLen({ "#1", "#3" }) [] normComponents, @IndexFor({ "#1" }) int normOffset, int @SameLen({ "#1", "#3" }) [] components, @IndexFor({ "#3" }) int offset);

    @Positive
    public float @SameLen({ "#1", "#3" }) [] getNormalizedComponents(int @SameLen({ "#1", "#3" }) [] components, @IndexFor({ "#1" }) int offset, float @SameLen({ "#1", "#3" }) [] normComponents, @IndexFor({ "#3" }) int normOffset);

    @Positive
    public int getDataElement(int[] components, @IndexFor({ "#1" }) int offset);

    @Positive
    public Object getDataElements(int[] components, @IndexFor({ "#1" }) int offset, Object obj);

    @Positive
    public int getDataElement(float[] normComponents, @IndexFor({ "#1" }) int normOffset);

    @Positive
    public Object getDataElements(float[] normComponents, @IndexFor({ "#1" }) int normOffset, Object obj);

    @Positive
    public float[] getNormalizedComponents(Object pixel, float[] normComponents, @IndexFor({ "#2" }) int normOffset);

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
    public final ColorSpace getColorSpace();

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
    @Deprecated()
    @Positive
    @SuppressWarnings("removal")
    @Positive
    public void finalize();

    @Positive
    public WritableRaster getAlphaRaster(WritableRaster raster);

    @Positive
    public String toString();

    @Positive
    static int getDefaultTransferType(int pixel_bits);

    @Positive
    static boolean isLinearRGBspace(ColorSpace cs);

    @Positive
    static boolean isLinearGRAYspace(ColorSpace cs);

    @Positive
    static byte[] getLinearRGB8TosRGB8LUT();

    @Positive
    static byte[] getsRGB8ToLinearRGB8LUT();

    @Positive
    static byte[] getLinearRGB16TosRGB8LUT();

    @Positive
    static short[] getsRGB8ToLinearRGB16LUT();

    @Positive
    static byte[] getGray8TosRGB8LUT(ICC_ColorSpace grayCS);

    @Positive
    static byte[] getLinearGray16ToOtherGray8LUT(ICC_ColorSpace grayCS);

    @Positive
    static byte[] getGray16TosRGB8LUT(ICC_ColorSpace grayCS);

    @Positive
    static short[] getLinearGray16ToOtherGray16LUT(ICC_ColorSpace grayCS);
    @Positive
}

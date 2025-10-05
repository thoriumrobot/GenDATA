/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package sun.java2d.loops;

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
import java.awt.image.ColorModel;
    @Positive
import sun.awt.image.PixelConverter;
    @Positive
import java.util.HashMap;

    @Positive
public final class SurfaceType {

    @Positive
    public static final String DESC_ANY;

    @Positive
    public static final String DESC_INT_RGB;

    @Positive
    public static final String DESC_INT_ARGB;

    @Positive
    public static final String DESC_INT_ARGB_PRE;

    @Positive
    public static final String DESC_INT_BGR;

    @Positive
    public static final String DESC_3BYTE_BGR;

    @Positive
    public static final String DESC_4BYTE_ABGR;

    @Positive
    public static final String DESC_4BYTE_ABGR_PRE;

    @Positive
    public static final String DESC_USHORT_565_RGB;

    @Positive
    public static final String DESC_USHORT_555_RGB;

    @Positive
    public static final String DESC_USHORT_555_RGBx;

    @Positive
    public static final String DESC_USHORT_4444_ARGB;

    @Positive
    public static final String DESC_BYTE_GRAY;

    @Positive
    public static final String DESC_USHORT_INDEXED;

    @Positive
    public static final String DESC_USHORT_GRAY;

    @Positive
    public static final String DESC_BYTE_BINARY;

    @Positive
    public static final String DESC_BYTE_INDEXED;

    @Positive
    public static final String DESC_ANY_INT;

    @Positive
    public static final String DESC_ANY_SHORT;

    @Positive
    public static final String DESC_ANY_BYTE;

    @Positive
    public static final String DESC_ANY_3BYTE;

    @Positive
    public static final String DESC_ANY_4BYTE;

    @Positive
    public static final String DESC_ANY_INT_DCM;

    @Positive
    public static final String DESC_INT_RGBx;

    @Positive
    public static final String DESC_INT_BGRx;

    @Positive
    public static final String DESC_3BYTE_RGB;

    @Positive
    public static final String DESC_INT_ARGB_BM;

    @Positive
    public static final String DESC_BYTE_INDEXED_BM;

    @Positive
    public static final String DESC_BYTE_INDEXED_OPAQUE;

    @Positive
    public static final String DESC_INDEX8_GRAY;

    @Positive
    public static final String DESC_INDEX12_GRAY;

    @Positive
    public static final String DESC_BYTE_BINARY_1BIT;

    @Positive
    public static final String DESC_BYTE_BINARY_2BIT;

    @Positive
    public static final String DESC_BYTE_BINARY_4BIT;

    @Positive
    public static final String DESC_ANY_PAINT;

    @Positive
    public static final String DESC_ANY_COLOR;

    @Positive
    public static final String DESC_OPAQUE_COLOR;

    @Positive
    public static final String DESC_GRADIENT_PAINT;

    @Positive
    public static final String DESC_OPAQUE_GRADIENT_PAINT;

    @Positive
    public static final String DESC_TEXTURE_PAINT;

    @Positive
    public static final String DESC_OPAQUE_TEXTURE_PAINT;

    @Positive
    public static final String DESC_LINEAR_GRADIENT_PAINT;

    @Positive
    public static final String DESC_OPAQUE_LINEAR_GRADIENT_PAINT;

    @Positive
    public static final String DESC_RADIAL_GRADIENT_PAINT;

    @Positive
    public static final String DESC_OPAQUE_RADIAL_GRADIENT_PAINT;

    @Positive
    public static final SurfaceType Any;

    @Positive
    public static final SurfaceType AnyInt;

    @Positive
    public static final SurfaceType AnyShort;

    @Positive
    public static final SurfaceType AnyByte;

    @Positive
    public static final SurfaceType AnyByteBinary;

    @Positive
    public static final SurfaceType Any3Byte;

    @Positive
    public static final SurfaceType Any4Byte;

    @Positive
    public static final SurfaceType AnyDcm;

    @Positive
    public static final SurfaceType Custom;

    @Positive
    public static final SurfaceType IntRgb;

    @Positive
    public static final SurfaceType IntArgb;

    @Positive
    public static final SurfaceType IntArgbPre;

    @Positive
    public static final SurfaceType IntBgr;

    @Positive
    public static final SurfaceType ThreeByteBgr;

    @Positive
    public static final SurfaceType FourByteAbgr;

    @Positive
    public static final SurfaceType FourByteAbgrPre;

    @Positive
    public static final SurfaceType Ushort565Rgb;

    @Positive
    public static final SurfaceType Ushort555Rgb;

    @Positive
    public static final SurfaceType Ushort555Rgbx;

    @Positive
    public static final SurfaceType Ushort4444Argb;

    @Positive
    public static final SurfaceType UshortIndexed;

    @Positive
    public static final SurfaceType ByteGray;

    @Positive
    public static final SurfaceType UshortGray;

    @Positive
    public static final SurfaceType ByteBinary1Bit;

    @Positive
    public static final SurfaceType ByteBinary2Bit;

    @Positive
    public static final SurfaceType ByteBinary4Bit;

    @Positive
    public static final SurfaceType ByteIndexed;

    @Positive
    public static final SurfaceType IntRgbx;

    @Positive
    public static final SurfaceType IntBgrx;

    @Positive
    public static final SurfaceType ThreeByteRgb;

    @Positive
    public static final SurfaceType IntArgbBm;

    @Positive
    public static final SurfaceType ByteIndexedBm;

    @Positive
    public static final SurfaceType ByteIndexedOpaque;

    @Positive
    public static final SurfaceType Index8Gray;

    @Positive
    public static final SurfaceType Index12Gray;

    @Positive
    public static final SurfaceType AnyPaint;

    @Positive
    public static final SurfaceType AnyColor;

    @Positive
    public static final SurfaceType OpaqueColor;

    @Positive
    public static final SurfaceType GradientPaint;

    @Positive
    public static final SurfaceType OpaqueGradientPaint;

    @Positive
    public static final SurfaceType LinearGradientPaint;

    @Positive
    public static final SurfaceType OpaqueLinearGradientPaint;

    @Positive
    public static final SurfaceType RadialGradientPaint;

    @Positive
    public static final SurfaceType OpaqueRadialGradientPaint;

    @Positive
    public static final SurfaceType TexturePaint;

    @Positive
    public static final SurfaceType OpaqueTexturePaint;

    @Positive
    public SurfaceType deriveSubType(String desc);

    @Positive
    public SurfaceType deriveSubType(String desc, PixelConverter pixelConverter);

    @Positive
    protected PixelConverter pixelConverter;

    @Positive
    public static synchronized int makeUniqueID(String desc);

    @Positive
    public int getUniqueID();

    @Positive
    public String getDescriptor();

    @Positive
    public SurfaceType getSuperType();

    @Positive
    public PixelConverter getPixelConverter();

    @Positive
    public int pixelFor(int rgb, ColorModel cm);

    @Positive
    public int rgbFor(int pixel, ColorModel cm);

    @Positive
    public int getAlphaMask();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public String toString();
    @Positive
}

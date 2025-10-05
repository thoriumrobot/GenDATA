/*
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package sun.java2d.loops;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.image.ColorModel;
import sun.awt.image.PixelConverter;
import java.util.HashMap;

public final class SurfaceType {

    public static final String DESC_ANY;

    public static final String DESC_INT_RGB;

    public static final String DESC_INT_ARGB;

    public static final String DESC_INT_ARGB_PRE;

    public static final String DESC_INT_BGR;

    public static final String DESC_3BYTE_BGR;

    public static final String DESC_4BYTE_ABGR;

    public static final String DESC_4BYTE_ABGR_PRE;

    public static final String DESC_USHORT_565_RGB;

    public static final String DESC_USHORT_555_RGB;

    public static final String DESC_USHORT_555_RGBx;

    public static final String DESC_USHORT_4444_ARGB;

    public static final String DESC_BYTE_GRAY;

    public static final String DESC_USHORT_INDEXED;

    public static final String DESC_USHORT_GRAY;

    public static final String DESC_BYTE_BINARY;

    public static final String DESC_BYTE_INDEXED;

    public static final String DESC_ANY_INT;

    public static final String DESC_ANY_SHORT;

    public static final String DESC_ANY_BYTE;

    public static final String DESC_ANY_3BYTE;

    public static final String DESC_ANY_4BYTE;

    public static final String DESC_ANY_INT_DCM;

    public static final String DESC_INT_RGBx;

    public static final String DESC_INT_BGRx;

    public static final String DESC_3BYTE_RGB;

    public static final String DESC_INT_ARGB_BM;

    public static final String DESC_BYTE_INDEXED_BM;

    public static final String DESC_BYTE_INDEXED_OPAQUE;

    public static final String DESC_INDEX8_GRAY;

    public static final String DESC_INDEX12_GRAY;

    public static final String DESC_BYTE_BINARY_1BIT;

    public static final String DESC_BYTE_BINARY_2BIT;

    public static final String DESC_BYTE_BINARY_4BIT;

    public static final String DESC_ANY_PAINT;

    public static final String DESC_ANY_COLOR;

    public static final String DESC_OPAQUE_COLOR;

    public static final String DESC_GRADIENT_PAINT;

    public static final String DESC_OPAQUE_GRADIENT_PAINT;

    public static final String DESC_TEXTURE_PAINT;

    public static final String DESC_OPAQUE_TEXTURE_PAINT;

    public static final String DESC_LINEAR_GRADIENT_PAINT;

    public static final String DESC_OPAQUE_LINEAR_GRADIENT_PAINT;

    public static final String DESC_RADIAL_GRADIENT_PAINT;

    public static final String DESC_OPAQUE_RADIAL_GRADIENT_PAINT;

    public static final SurfaceType Any;

    public static final SurfaceType AnyInt;

    public static final SurfaceType AnyShort;

    public static final SurfaceType AnyByte;

    public static final SurfaceType AnyByteBinary;

    public static final SurfaceType Any3Byte;

    public static final SurfaceType Any4Byte;

    public static final SurfaceType AnyDcm;

    public static final SurfaceType Custom;

    public static final SurfaceType IntRgb;

    public static final SurfaceType IntArgb;

    public static final SurfaceType IntArgbPre;

    public static final SurfaceType IntBgr;

    public static final SurfaceType ThreeByteBgr;

    public static final SurfaceType FourByteAbgr;

    public static final SurfaceType FourByteAbgrPre;

    public static final SurfaceType Ushort565Rgb;

    public static final SurfaceType Ushort555Rgb;

    public static final SurfaceType Ushort555Rgbx;

    public static final SurfaceType Ushort4444Argb;

    public static final SurfaceType UshortIndexed;

    public static final SurfaceType ByteGray;

    public static final SurfaceType UshortGray;

    public static final SurfaceType ByteBinary1Bit;

    public static final SurfaceType ByteBinary2Bit;

    public static final SurfaceType ByteBinary4Bit;

    public static final SurfaceType ByteIndexed;

    public static final SurfaceType IntRgbx;

    public static final SurfaceType IntBgrx;

    public static final SurfaceType ThreeByteRgb;

    public static final SurfaceType IntArgbBm;

    public static final SurfaceType ByteIndexedBm;

    public static final SurfaceType ByteIndexedOpaque;

    public static final SurfaceType Index8Gray;

    public static final SurfaceType Index12Gray;

    public static final SurfaceType AnyPaint;

    public static final SurfaceType AnyColor;

    public static final SurfaceType OpaqueColor;

    public static final SurfaceType GradientPaint;

    public static final SurfaceType OpaqueGradientPaint;

    public static final SurfaceType LinearGradientPaint;

    public static final SurfaceType OpaqueLinearGradientPaint;

    public static final SurfaceType RadialGradientPaint;

    public static final SurfaceType OpaqueRadialGradientPaint;

    public static final SurfaceType TexturePaint;

    public static final SurfaceType OpaqueTexturePaint;

    public SurfaceType deriveSubType(String desc);

    public SurfaceType deriveSubType(String desc, PixelConverter pixelConverter);

    protected PixelConverter pixelConverter;

    public static synchronized int makeUniqueID(String desc);

    public int getUniqueID();

    public String getDescriptor();

    public SurfaceType getSuperType();

    public PixelConverter getPixelConverter();

    public int pixelFor(int rgb, ColorModel cm);

    public int rgbFor(int pixel, ColorModel cm);

    public int getAlphaMask();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public String toString();
}

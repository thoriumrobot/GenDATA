/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.imageio;

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
import java.awt.Point;
    @Positive
import java.awt.Transparency;
    @Positive
import java.awt.image.BandedSampleModel;
    @Positive
import java.awt.image.BufferedImage;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.color.ColorSpace;
    @Positive
import java.awt.image.IndexColorModel;
    @Positive
import java.awt.image.ComponentColorModel;
    @Positive
import java.awt.image.DataBuffer;
    @Positive
import java.awt.image.DirectColorModel;
    @Positive
import java.awt.image.MultiPixelPackedSampleModel;
    @Positive
import java.awt.image.PixelInterleavedSampleModel;
    @Positive
import java.awt.image.SinglePixelPackedSampleModel;
    @Positive
import java.awt.image.Raster;
    @Positive
import java.awt.image.RenderedImage;
    @Positive
import java.awt.image.SampleModel;
    @Positive
import java.awt.image.WritableRaster;
    @Positive
import java.util.Hashtable;

    @Positive
public class ImageTypeSpecifier {

    @Positive
    protected ColorModel colorModel;

    @Positive
    protected SampleModel sampleModel;

    @Positive
    public ImageTypeSpecifier(ColorModel colorModel, SampleModel sampleModel) {
    @Positive
    }

    @Positive
    public ImageTypeSpecifier(RenderedImage image) {
    @Positive
    }

    @Positive
    static class Packed extends ImageTypeSpecifier {

    @Positive
        public Packed(ColorSpace colorSpace, int redMask, int greenMask, int blueMask, int alphaMask, int transferType, boolean isAlphaPremultiplied) {
    @Positive
        }
    @Positive
    }

    @Positive
    public static ImageTypeSpecifier createPacked(ColorSpace colorSpace, int redMask, int greenMask, int blueMask, int alphaMask, int transferType, boolean isAlphaPremultiplied);

    @Positive
    static ColorModel createComponentCM(ColorSpace colorSpace, int numBands, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied);

    @Positive
    static class Interleaved extends ImageTypeSpecifier {

    @Positive
        public Interleaved(ColorSpace colorSpace, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied) {
    @Positive
        }

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static ImageTypeSpecifier createInterleaved(ColorSpace colorSpace, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied);

    @Positive
    static class Banded extends ImageTypeSpecifier {

    @Positive
        public Banded(ColorSpace colorSpace, int[] bankIndices, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied) {
    @Positive
        }

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static ImageTypeSpecifier createBanded(ColorSpace colorSpace, int[] bankIndices, int[] bandOffsets, int dataType, boolean hasAlpha, boolean isAlphaPremultiplied);

    @Positive
    static class Grayscale extends ImageTypeSpecifier {

    @Positive
        public Grayscale(int bits, int dataType, boolean isSigned, boolean hasAlpha, boolean isAlphaPremultiplied) {
    @Positive
        }
    @Positive
    }

    @Positive
    public static ImageTypeSpecifier createGrayscale(int bits, int dataType, boolean isSigned);

    @Positive
    public static ImageTypeSpecifier createGrayscale(int bits, int dataType, boolean isSigned, boolean isAlphaPremultiplied);

    @Positive
    static class Indexed extends ImageTypeSpecifier {

    @Positive
        public Indexed(byte[] redLUT, byte[] greenLUT, byte[] blueLUT, byte[] alphaLUT, int bits, int dataType) {
    @Positive
        }
    @Positive
    }

    @Positive
    public static ImageTypeSpecifier createIndexed(byte[] redLUT, byte[] greenLUT, byte[] blueLUT, byte[] alphaLUT, int bits, int dataType);

    @Positive
    public static ImageTypeSpecifier createFromBufferedImageType(int bufferedImageType);

    @Positive
    public static ImageTypeSpecifier createFromRenderedImage(RenderedImage image);

    @Positive
    public int getBufferedImageType();

    @Positive
    public int getNumComponents();

    @Positive
    public int getNumBands();

    @Positive
    public int getBitsPerBand(int band);

    @Positive
    public SampleModel getSampleModel();

    @Positive
    public SampleModel getSampleModel(int width, int height);

    @Positive
    public ColorModel getColorModel();

    @Positive
    public BufferedImage createBufferedImage(int width, int height);

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

/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.image.BufferedImage;
    @Positive
import java.awt.image.Raster;
    @Positive
import java.awt.image.WritableRaster;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.awt.image.DirectColorModel;
    @Positive
import java.awt.image.IndexColorModel;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.NoninvertibleTransformException;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import sun.awt.image.SunWritableRaster;
    @Positive
import sun.awt.image.IntegerInterleavedRaster;
    @Positive
import sun.awt.image.ByteInterleavedRaster;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
abstract class TexturePaintContext implements PaintContext {

    @Positive
    public static ColorModel xrgbmodel;

    @Positive
    public static ColorModel argbmodel;

    @Positive
    public static PaintContext getContext(BufferedImage bufImg, AffineTransform xform, RenderingHints hints, Rectangle devBounds);

    @Positive
    public static boolean isFilterableICM(ColorModel cm);

    @Positive
    public static boolean isFilterableDCM(ColorModel cm);

    @Positive
    public static boolean isMaskOK(int mask, boolean canbezero);

    @Positive
    public static ColorModel getInternedColorModel(ColorModel cm);

    @Positive
    static int fractAsInt(double d);

    @Positive
    static double mod(double num, double den);

    @Positive
    public void dispose();

    @Positive
    public ColorModel getColorModel();

    @Positive
    public Raster getRaster(int x, int y, int w, int h);

    @Positive
    static synchronized WritableRaster makeRaster(ColorModel cm, Raster srcRas, int w, int h);

    @Positive
    static synchronized void dropRaster(ColorModel cm, Raster outRas);

    @Positive
    static synchronized WritableRaster makeByteRaster(Raster srcRas, int w, int h);

    @Positive
    static synchronized void dropByteRaster(Raster outRas);

    @Positive
    public abstract WritableRaster makeRaster(int w, int h);

    @Positive
    public abstract void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);

    @Positive
    public static int blend(int[] rgbs, int xmul, int ymul);

    @Positive
    static class Int extends TexturePaintContext {

    @Positive
        public Int(IntegerInterleavedRaster srcRas, ColorModel cm, AffineTransform xform, int maxw, boolean filter) {
    @Positive
        }

    @Positive
        public WritableRaster makeRaster(int w, int h);

    @Positive
        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    @Positive
    }

    @Positive
    static class Byte extends TexturePaintContext {

    @Positive
        public Byte(ByteInterleavedRaster srcRas, ColorModel cm, AffineTransform xform, int maxw) {
    @Positive
        }

    @Positive
        public WritableRaster makeRaster(int w, int h);

    @Positive
        public void dispose();

    @Positive
        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    @Positive
    }

    @Positive
    static class ByteFilter extends TexturePaintContext {

    @Positive
        public ByteFilter(ByteInterleavedRaster srcRas, ColorModel cm, AffineTransform xform, int maxw) {
    @Positive
        }

    @Positive
        public WritableRaster makeRaster(int w, int h);

    @Positive
        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    @Positive
    }

    @Positive
    static class Any extends TexturePaintContext {

    @Positive
        public Any(WritableRaster srcRas, ColorModel cm, AffineTransform xform, int maxw, boolean filter) {
    @Positive
        }

    @Positive
        public WritableRaster makeRaster(int w, int h);

    @Positive
        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    @Positive
    }
    @Positive
}

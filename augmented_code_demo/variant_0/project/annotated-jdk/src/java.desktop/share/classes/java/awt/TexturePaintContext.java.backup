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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.awt.image.ColorModel;
import java.awt.image.DirectColorModel;
import java.awt.image.IndexColorModel;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.lang.ref.WeakReference;
import sun.awt.image.SunWritableRaster;
import sun.awt.image.IntegerInterleavedRaster;
import sun.awt.image.ByteInterleavedRaster;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
abstract class TexturePaintContext implements PaintContext {

    public static ColorModel xrgbmodel;

    public static ColorModel argbmodel;

    public static PaintContext getContext(BufferedImage bufImg, AffineTransform xform, RenderingHints hints, Rectangle devBounds);

    public static boolean isFilterableICM(ColorModel cm);

    public static boolean isFilterableDCM(ColorModel cm);

    public static boolean isMaskOK(int mask, boolean canbezero);

    public static ColorModel getInternedColorModel(ColorModel cm);

    static int fractAsInt(double d);

    static double mod(double num, double den);

    public void dispose();

    public ColorModel getColorModel();

    public Raster getRaster(int x, int y, int w, int h);

    static synchronized WritableRaster makeRaster(ColorModel cm, Raster srcRas, int w, int h);

    static synchronized void dropRaster(ColorModel cm, Raster outRas);

    static synchronized WritableRaster makeByteRaster(Raster srcRas, int w, int h);

    static synchronized void dropByteRaster(Raster outRas);

    public abstract WritableRaster makeRaster(int w, int h);

    public abstract void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);

    public static int blend(int[] rgbs, int xmul, int ymul);

    static class Int extends TexturePaintContext {

        public Int(IntegerInterleavedRaster srcRas, ColorModel cm, AffineTransform xform, int maxw, boolean filter) {
        }

        public WritableRaster makeRaster(int w, int h);

        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    }

    static class Byte extends TexturePaintContext {

        public Byte(ByteInterleavedRaster srcRas, ColorModel cm, AffineTransform xform, int maxw) {
        }

        public WritableRaster makeRaster(int w, int h);

        public void dispose();

        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    }

    static class ByteFilter extends TexturePaintContext {

        public ByteFilter(ByteInterleavedRaster srcRas, ColorModel cm, AffineTransform xform, int maxw) {
        }

        public WritableRaster makeRaster(int w, int h);

        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    }

    static class Any extends TexturePaintContext {

        public Any(WritableRaster srcRas, ColorModel cm, AffineTransform xform, int maxw, boolean filter) {
        }

        public WritableRaster makeRaster(int w, int h);

        public void setRaster(int x, int y, int xerr, int yerr, int w, int h, int bWidth, int bHeight, int colincx, int colincxerr, int colincy, int colincyerr, int rowincx, int rowincxerr, int rowincy, int rowincyerr);
    }
}

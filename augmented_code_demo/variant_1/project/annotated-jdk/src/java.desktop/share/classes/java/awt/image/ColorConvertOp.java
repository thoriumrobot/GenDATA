/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Graphics2D;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.RenderingHints;
    @Positive
import java.awt.color.ColorSpace;
    @Positive
import java.awt.color.ICC_ColorSpace;
    @Positive
import java.awt.color.ICC_Profile;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import sun.java2d.cmm.CMSManager;
    @Positive
import sun.java2d.cmm.ColorTransform;
    @Positive
import sun.java2d.cmm.PCMM;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class ColorConvertOp implements BufferedImageOp, RasterOp {

    @Positive
    public ColorConvertOp(RenderingHints hints) {
    @Positive
    }

    @Positive
    public ColorConvertOp(ColorSpace cspace, RenderingHints hints) {
    @Positive
    }

    @Positive
    public ColorConvertOp(ColorSpace srcCspace, ColorSpace dstCspace, RenderingHints hints) {
    @Positive
    }

    @Positive
    public ColorConvertOp(ICC_Profile[] profiles, RenderingHints hints) {
    @Positive
    }

    @Positive
    public final ICC_Profile[] getICC_Profiles();

    @Positive
    public final BufferedImage filter(BufferedImage src, BufferedImage dest);

    @Positive
    public final WritableRaster filter(Raster src, WritableRaster dest);

    @Positive
    public final Rectangle2D getBounds2D(BufferedImage src);

    @Positive
    public final Rectangle2D getBounds2D(Raster src);

    @Positive
    public BufferedImage createCompatibleDestImage(BufferedImage src, ColorModel destCM);

    @Positive
    public WritableRaster createCompatibleDestRaster(Raster src);

    @Positive
    public final Point2D getPoint2D(Point2D srcPt, Point2D dstPt);

    @Positive
    public final RenderingHints getRenderingHints();
    @Positive
}

// CFWR semantic augmentation - variant 1

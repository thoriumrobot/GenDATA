/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2014, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.awt.image;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.NoninvertibleTransformException;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.AlphaComposite;
    @Positive
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.RenderingHints;
    @Positive
import java.awt.Transparency;
    @Positive
import java.lang.annotation.Native;
    @Positive
import sun.awt.image.ImagingLib;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AffineTransformOp implements BufferedImageOp, RasterOp {

    @Positive
    @Native
    @Positive
    public static final int TYPE_NEAREST_NEIGHBOR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_BILINEAR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_BICUBIC;

    @Positive
    public AffineTransformOp(AffineTransform xform, RenderingHints hints) {
    @Positive
    }

    @Positive
    public AffineTransformOp(AffineTransform xform, int interpolationType) {
    @Positive
    }

    @Positive
    public final int getInterpolationType();

    @Positive
    public final BufferedImage filter(BufferedImage src, BufferedImage dst);

    @Positive
    public final WritableRaster filter(Raster src, WritableRaster dst);

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
    public final AffineTransform getTransform();

    @Positive
    public final RenderingHints getRenderingHints();

    @Positive
    void validateTransform(AffineTransform xform);
    @Positive
}

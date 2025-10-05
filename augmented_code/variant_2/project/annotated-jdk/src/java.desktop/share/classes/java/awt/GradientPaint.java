/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2010, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.beans.ConstructorProperties;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class GradientPaint implements Paint {

    @Positive
    public GradientPaint(float x1, float y1, Color color1, float x2, float y2, Color color2) {
    @Positive
    }

    @Positive
    public GradientPaint(Point2D pt1, Color color1, Point2D pt2, Color color2) {
    @Positive
    }

    @Positive
    public GradientPaint(float x1, float y1, Color color1, float x2, float y2, Color color2, boolean cyclic) {
    @Positive
    }

    @Positive
    @ConstructorProperties({ "point1", "color1", "point2", "color2", "cyclic" })
    @Positive
    public GradientPaint(Point2D pt1, Color color1, Point2D pt2, Color color2, boolean cyclic) {
    @Positive
    }

    @Positive
    public Point2D getPoint1();

    @Positive
    public Color getColor1();

    @Positive
    public Point2D getPoint2();

    @Positive
    public Color getColor2();

    @Positive
    public boolean isCyclic();

    @Positive
    public PaintContext createContext(ColorModel cm, Rectangle deviceBounds, Rectangle2D userBounds, AffineTransform xform, RenderingHints hints);

    @Positive
    public int getTransparency();
    @Positive
}

/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.geom;

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
import java.io.Serial;
    @Positive
import java.io.Serializable;

    @Positive
public abstract class Ellipse2D extends RectangularShape {

    @Positive
    public static class Float extends Ellipse2D implements Serializable {

    @Positive
        public float x;

    @Positive
        public float y;

    @Positive
        public float width;

    @Positive
        public float height;

    @Positive
        public Float() {
    @Positive
        }

    @Positive
        public Float(float x, float y, float w, float h) {
    @Positive
        }

    @Positive
        public double getX();

    @Positive
        public double getY();

    @Positive
        public double getWidth();

    @Positive
        public double getHeight();

    @Positive
        public boolean isEmpty();

    @Positive
        public void setFrame(float x, float y, float w, float h);

    @Positive
        public void setFrame(double x, double y, double w, double h);

    @Positive
        public Rectangle2D getBounds2D();
    @Positive
    }

    @Positive
    public static class Double extends Ellipse2D implements Serializable {

    @Positive
        public double x;

    @Positive
        public double y;

    @Positive
        public double width;

    @Positive
        public double height;

    @Positive
        public Double() {
    @Positive
        }

    @Positive
        public Double(double x, double y, double w, double h) {
    @Positive
        }

    @Positive
        public double getX();

    @Positive
        public double getY();

    @Positive
        public double getWidth();

    @Positive
        public double getHeight();

    @Positive
        public boolean isEmpty();

    @Positive
        public void setFrame(double x, double y, double w, double h);

    @Positive
        public Rectangle2D getBounds2D();
    @Positive
    }

    @Positive
    protected Ellipse2D() {
    @Positive
    }

    @Positive
    public boolean contains(double x, double y);

    @Positive
    public boolean intersects(double x, double y, double w, double h);

    @Positive
    public boolean contains(double x, double y, double w, double h);

    @Positive
    public PathIterator getPathIterator(AffineTransform at);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);
    @Positive
}

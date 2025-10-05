/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.geom.Dimension2D;
    @Positive
import java.beans.Transient;
    @Positive
import java.io.Serial;

    @Positive
public class Dimension extends Dimension2D implements java.io.Serializable {

    @Positive
    public int width;

    @Positive
    public int height;

    @Positive
    public Dimension() {
    @Positive
    }

    @Positive
    public Dimension(Dimension d) {
    @Positive
    }

    @Positive
    public Dimension(int width, int height) {
    @Positive
    }

    @Positive
    public double getWidth();

    @Positive
    public double getHeight();

    @Positive
    public void setSize(double width, double height);

    @Positive
    @Transient
    @Positive
    public Dimension getSize();

    @Positive
    public void setSize(Dimension d);

    @Positive
    public void setSize(int width, int height);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();
    @Positive
}

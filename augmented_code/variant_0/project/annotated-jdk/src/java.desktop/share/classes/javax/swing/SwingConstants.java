/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2000, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

    @Positive
import org.checkerframework.checker.fenum.qual.*;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;

    @Positive
@AnnotatedFor("fenum")
    @Positive
public interface SwingConstants {

    @Positive
    @SwingCompassDirection
    @Positive
    @SwingHorizontalOrientation
    @Positive
    @SwingVerticalOrientation
    @Positive
    public static final int CENTER;

    @Positive
    @SwingVerticalOrientation
    @Positive
    public static final int TOP;

    @Positive
    @SwingHorizontalOrientation
    @Positive
    public static final int LEFT;

    @Positive
    @SwingVerticalOrientation
    @Positive
    public static final int BOTTOM;

    @Positive
    @SwingHorizontalOrientation
    @Positive
    public static final int RIGHT;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int NORTH;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int NORTH_EAST;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int EAST;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int SOUTH_EAST;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int SOUTH;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int SOUTH_WEST;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int WEST;

    @Positive
    @SwingCompassDirection
    @Positive
    public static final int NORTH_WEST;

    @Positive
    @SwingElementOrientation
    @Positive
    public static final int HORIZONTAL;

    @Positive
    @SwingElementOrientation
    @Positive
    public static final int VERTICAL;

    @Positive
    @SwingHorizontalOrientation
    @Positive
    @SwingTextOrientation
    @Positive
    public static final int LEADING;

    @Positive
    @SwingHorizontalOrientation
    @Positive
    @SwingTextOrientation
    @Positive
    public static final int TRAILING;

    @Positive
    @SwingTextOrientation
    @Positive
    public static final int NEXT;

    @Positive
    @SwingTextOrientation
    @Positive
    public static final int PREVIOUS;
    @Positive
}

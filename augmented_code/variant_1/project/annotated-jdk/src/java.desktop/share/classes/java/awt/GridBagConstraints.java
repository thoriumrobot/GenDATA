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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.Serial;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class GridBagConstraints implements Cloneable, java.io.Serializable {

    @Positive
    public static final int RELATIVE;

    @Positive
    public static final int REMAINDER;

    @Positive
    public static final int NONE;

    @Positive
    public static final int BOTH;

    @Positive
    public static final int HORIZONTAL;

    @Positive
    public static final int VERTICAL;

    @Positive
    public static final int CENTER;

    @Positive
    public static final int NORTH;

    @Positive
    public static final int NORTHEAST;

    @Positive
    public static final int EAST;

    @Positive
    public static final int SOUTHEAST;

    @Positive
    public static final int SOUTH;

    @Positive
    public static final int SOUTHWEST;

    @Positive
    public static final int WEST;

    @Positive
    public static final int NORTHWEST;

    @Positive
    public static final int PAGE_START;

    @Positive
    public static final int PAGE_END;

    @Positive
    public static final int LINE_START;

    @Positive
    public static final int LINE_END;

    @Positive
    public static final int FIRST_LINE_START;

    @Positive
    public static final int FIRST_LINE_END;

    @Positive
    public static final int LAST_LINE_START;

    @Positive
    public static final int LAST_LINE_END;

    @Positive
    public static final int BASELINE;

    @Positive
    public static final int BASELINE_LEADING;

    @Positive
    public static final int BASELINE_TRAILING;

    @Positive
    public static final int ABOVE_BASELINE;

    @Positive
    public static final int ABOVE_BASELINE_LEADING;

    @Positive
    public static final int ABOVE_BASELINE_TRAILING;

    @Positive
    public static final int BELOW_BASELINE;

    @Positive
    public static final int BELOW_BASELINE_LEADING;

    @Positive
    public static final int BELOW_BASELINE_TRAILING;

    @Positive
    public int gridx;

    @Positive
    public int gridy;

    @Positive
    public int gridwidth;

    @Positive
    public int gridheight;

    @Positive
    public double weightx;

    @Positive
    public double weighty;

    @Positive
    public int anchor;

    @Positive
    public int fill;

    @Positive
    public Insets insets;

    @Positive
    public int ipadx;

    @Positive
    public int ipady;

    @Positive
    public GridBagConstraints() {
    @Positive
    }

    @Positive
    public GridBagConstraints(int gridx, int gridy, int gridwidth, int gridheight, double weightx, double weighty, int anchor, int fill, Insets insets, int ipadx, int ipady) {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    boolean isVerticallyResizable();
    @Positive
}

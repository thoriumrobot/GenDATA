/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

// CFWR semantic augmentation - variant 0

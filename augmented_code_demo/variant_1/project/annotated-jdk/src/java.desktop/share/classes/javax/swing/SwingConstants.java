/*
    @Positive
 * Copyright (c) 1997, 2000, Oracle and/or its affiliates. All rights reserved.
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

// CFWR semantic augmentation - variant 1

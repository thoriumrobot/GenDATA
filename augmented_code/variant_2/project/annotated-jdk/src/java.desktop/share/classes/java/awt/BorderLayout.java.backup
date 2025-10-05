/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.Serial;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class BorderLayout implements LayoutManager2, java.io.Serializable {

    @Interned
    public static final String NORTH;

    @Interned
    public static final String SOUTH;

    @Interned
    public static final String EAST;

    @Interned
    public static final String WEST;

    @Interned
    public static final String CENTER;

    @Interned
    public static final String BEFORE_FIRST_LINE;

    @Interned
    public static final String AFTER_LAST_LINE;

    @Interned
    public static final String BEFORE_LINE_BEGINS;

    @Interned
    public static final String AFTER_LINE_ENDS;

    public static final String PAGE_START;

    public static final String PAGE_END;

    public static final String LINE_START;

    public static final String LINE_END;

    public BorderLayout() {
    }

    public BorderLayout(int hgap, int vgap) {
    }

    public int getHgap();

    public void setHgap(int hgap);

    public int getVgap();

    public void setVgap(int vgap);

    public void addLayoutComponent(Component comp, Object constraints);

    @Deprecated
    public void addLayoutComponent(String name, Component comp);

    public void removeLayoutComponent(Component comp);

    public Component getLayoutComponent(Object constraints);

    public Component getLayoutComponent(Container target, Object constraints);

    public Object getConstraints(Component comp);

    public Dimension minimumLayoutSize(Container target);

    public Dimension preferredLayoutSize(Container target);

    public Dimension maximumLayoutSize(Container target);

    public float getLayoutAlignmentX(Container parent);

    public float getLayoutAlignmentY(Container parent);

    public void invalidateLayout(Container target);

    public void layoutContainer(Container target);

    public String toString();
}

/*
 * Copyright (c) 2001, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Insets;
import java.awt.LayoutManager2;
import java.awt.Rectangle;
import java.util.*;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class SpringLayout implements LayoutManager2 {

    @Interned
    public static final String NORTH;

    @Interned
    public static final String SOUTH;

    @Interned
    public static final String EAST;

    @Interned
    public static final String WEST;

    @Interned
    public static final String HORIZONTAL_CENTER;

    @Interned
    public static final String VERTICAL_CENTER;

    @Interned
    public static final String BASELINE;

    @Interned
    public static final String WIDTH;

    @Interned
    public static final String HEIGHT;

    public static class Constraints {

        public Constraints() {
        }

        public Constraints(Spring x, Spring y) {
        }

        public Constraints(Spring x, Spring y, Spring width, Spring height) {
        }

        public Constraints(Component c) {
        }

        public void setX(Spring x);

        public Spring getX();

        public void setY(Spring y);

        public Spring getY();

        public void setWidth(Spring width);

        public Spring getWidth();

        public void setHeight(Spring height);

        public Spring getHeight();

        public void setConstraint(String edgeName, Spring s);

        public Spring getConstraint(String edgeName);

        void reset();
    }

    private static class SpringProxy extends Spring {

        public SpringProxy(String edgeName, Component c, SpringLayout l) {
        }

        public int getMinimumValue();

        public int getPreferredValue();

        public int getMaximumValue();

        public int getValue();

        public void setValue(int size);

        boolean isCyclic(SpringLayout l);

        public String toString();
    }

    public SpringLayout() {
    }

    boolean isCyclic(Spring s);

    public void addLayoutComponent(String name, Component c);

    public void removeLayoutComponent(Component c);

    public Dimension minimumLayoutSize(Container parent);

    public Dimension preferredLayoutSize(Container parent);

    public Dimension maximumLayoutSize(Container parent);

    public void addLayoutComponent(Component component, Object constraints);

    public float getLayoutAlignmentX(Container p);

    public float getLayoutAlignmentY(Container p);

    public void invalidateLayout(Container p);

    public void putConstraint(String e1, Component c1, int pad, String e2, Component c2);

    public void putConstraint(String e1, Component c1, Spring s, String e2, Component c2);

    public Constraints getConstraints(Component c);

    public Spring getConstraint(String edgeName, Component c);

    public void layoutContainer(Container parent);
}

/*
    @Positive
 * Copyright (c) 2001, 2014, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.FontMetrics;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.LayoutManager2;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.util.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class SpringLayout implements LayoutManager2 {

    @Positive
    @Interned
    @Positive
    public static final String NORTH;

    @Positive
    @Interned
    @Positive
    public static final String SOUTH;

    @Positive
    @Interned
    @Positive
    public static final String EAST;

    @Positive
    @Interned
    @Positive
    public static final String WEST;

    @Positive
    @Interned
    @Positive
    public static final String HORIZONTAL_CENTER;

    @Positive
    @Interned
    @Positive
    public static final String VERTICAL_CENTER;

    @Positive
    @Interned
    @Positive
    public static final String BASELINE;

    @Positive
    @Interned
    @Positive
    public static final String WIDTH;

    @Positive
    @Interned
    @Positive
    public static final String HEIGHT;

    @Positive
    public static class Constraints {

    @Positive
        public Constraints() {
    @Positive
        }

    @Positive
        public Constraints(Spring x, Spring y) {
    @Positive
        }

    @Positive
        public Constraints(Spring x, Spring y, Spring width, Spring height) {
    @Positive
        }

    @Positive
        public Constraints(Component c) {
    @Positive
        }

    @Positive
        public void setX(Spring x);

    @Positive
        public Spring getX();

    @Positive
        public void setY(Spring y);

    @Positive
        public Spring getY();

    @Positive
        public void setWidth(Spring width);

    @Positive
        public Spring getWidth();

    @Positive
        public void setHeight(Spring height);

    @Positive
        public Spring getHeight();

    @Positive
        public void setConstraint(String edgeName, Spring s);

    @Positive
        public Spring getConstraint(String edgeName);

    @Positive
        void reset();
    @Positive
    }

    @Positive
    private static class SpringProxy extends Spring {

    @Positive
        public SpringProxy(String edgeName, Component c, SpringLayout l) {
    @Positive
        }

    @Positive
        public int getMinimumValue();

    @Positive
        public int getPreferredValue();

    @Positive
        public int getMaximumValue();

    @Positive
        public int getValue();

    @Positive
        public void setValue(int size);

    @Positive
        boolean isCyclic(SpringLayout l);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public SpringLayout() {
    @Positive
    }

    @Positive
    boolean isCyclic(Spring s);

    @Positive
    public void addLayoutComponent(String name, Component c);

    @Positive
    public void removeLayoutComponent(Component c);

    @Positive
    public Dimension minimumLayoutSize(Container parent);

    @Positive
    public Dimension preferredLayoutSize(Container parent);

    @Positive
    public Dimension maximumLayoutSize(Container parent);

    @Positive
    public void addLayoutComponent(Component component, Object constraints);

    @Positive
    public float getLayoutAlignmentX(Container p);

    @Positive
    public float getLayoutAlignmentY(Container p);

    @Positive
    public void invalidateLayout(Container p);

    @Positive
    public void putConstraint(String e1, Component c1, int pad, String e2, Component c2);

    @Positive
    public void putConstraint(String e1, Component c1, Spring s, String e2, Component c2);

    @Positive
    public Constraints getConstraints(Component c);

    @Positive
    public Spring getConstraint(String edgeName, Component c);

    @Positive
    public void layoutContainer(Container parent);
    @Positive
}

// CFWR semantic augmentation - variant 0

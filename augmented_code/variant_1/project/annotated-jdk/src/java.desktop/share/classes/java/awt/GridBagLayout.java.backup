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
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.Serial;
import java.util.Arrays;
import java.util.Hashtable;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class GridBagLayout implements LayoutManager2, java.io.Serializable {

    protected static final int MAXGRIDSIZE;

    protected static final int MINSIZE;

    protected static final int PREFERREDSIZE;

    protected Hashtable<Component, GridBagConstraints> comptable;

    protected GridBagConstraints defaultConstraints;

    protected GridBagLayoutInfo layoutInfo;

    public int[] columnWidths;

    public int[] rowHeights;

    public double[] columnWeights;

    public double[] rowWeights;

    public GridBagLayout() {
    }

    public void setConstraints(Component comp, GridBagConstraints constraints);

    public GridBagConstraints getConstraints(Component comp);

    protected GridBagConstraints lookupConstraints(Component comp);

    public Point getLayoutOrigin();

    public int[][] getLayoutDimensions();

    public double[][] getLayoutWeights();

    public Point location(int x, int y);

    public void addLayoutComponent(String name, Component comp);

    public void addLayoutComponent(Component comp, Object constraints);

    public void removeLayoutComponent(Component comp);

    public Dimension preferredLayoutSize(Container parent);

    public Dimension minimumLayoutSize(Container parent);

    public Dimension maximumLayoutSize(Container target);

    public float getLayoutAlignmentX(Container parent);

    public float getLayoutAlignmentY(Container parent);

    public void invalidateLayout(Container target);

    public void layoutContainer(Container parent);

    public String toString();

    protected GridBagLayoutInfo getLayoutInfo(Container parent, int sizeflag);

    protected GridBagLayoutInfo GetLayoutInfo(Container parent, int sizeflag);

    protected void adjustForGravity(GridBagConstraints constraints, Rectangle r);

    protected void AdjustForGravity(GridBagConstraints constraints, Rectangle r);

    protected Dimension getMinSize(Container parent, GridBagLayoutInfo info);

    protected Dimension GetMinSize(Container parent, GridBagLayoutInfo info);

    protected void arrangeGrid(Container parent);

    protected void ArrangeGrid(Container parent);
}

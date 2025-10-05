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
import java.util.Arrays;
    @Positive
import java.util.Hashtable;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class GridBagLayout implements LayoutManager2, java.io.Serializable {

    @Positive
    protected static final int MAXGRIDSIZE;

    @Positive
    protected static final int MINSIZE;

    @Positive
    protected static final int PREFERREDSIZE;

    @Positive
    protected Hashtable<Component, GridBagConstraints> comptable;

    @Positive
    protected GridBagConstraints defaultConstraints;

    @Positive
    protected GridBagLayoutInfo layoutInfo;

    @Positive
    public int[] columnWidths;

    @Positive
    public int[] rowHeights;

    @Positive
    public double[] columnWeights;

    @Positive
    public double[] rowWeights;

    @Positive
    public GridBagLayout() {
    @Positive
    }

    @Positive
    public void setConstraints(Component comp, GridBagConstraints constraints);

    @Positive
    public GridBagConstraints getConstraints(Component comp);

    @Positive
    protected GridBagConstraints lookupConstraints(Component comp);

    @Positive
    public Point getLayoutOrigin();

    @Positive
    public int[][] getLayoutDimensions();

    @Positive
    public double[][] getLayoutWeights();

    @Positive
    public Point location(int x, int y);

    @Positive
    public void addLayoutComponent(String name, Component comp);

    @Positive
    public void addLayoutComponent(Component comp, Object constraints);

    @Positive
    public void removeLayoutComponent(Component comp);

    @Positive
    public Dimension preferredLayoutSize(Container parent);

    @Positive
    public Dimension minimumLayoutSize(Container parent);

    @Positive
    public Dimension maximumLayoutSize(Container target);

    @Positive
    public float getLayoutAlignmentX(Container parent);

    @Positive
    public float getLayoutAlignmentY(Container parent);

    @Positive
    public void invalidateLayout(Container target);

    @Positive
    public void layoutContainer(Container parent);

    @Positive
    public String toString();

    @Positive
    protected GridBagLayoutInfo getLayoutInfo(Container parent, int sizeflag);

    @Positive
    protected GridBagLayoutInfo GetLayoutInfo(Container parent, int sizeflag);

    @Positive
    protected void adjustForGravity(GridBagConstraints constraints, Rectangle r);

    @Positive
    protected void AdjustForGravity(GridBagConstraints constraints, Rectangle r);

    @Positive
    protected Dimension getMinSize(Container parent, GridBagLayoutInfo info);

    @Positive
    protected Dimension GetMinSize(Container parent, GridBagLayoutInfo info);

    @Positive
    protected void arrangeGrid(Container parent);

    @Positive
    protected void ArrangeGrid(Container parent);
    @Positive
}

// CFWR semantic augmentation - variant 0

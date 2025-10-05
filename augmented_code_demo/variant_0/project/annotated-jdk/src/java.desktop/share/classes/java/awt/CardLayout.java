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
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Vector;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class CardLayout implements LayoutManager2, Serializable {

    @Positive
    class Card implements Serializable {

    @Positive
        public String name;

    @Positive
        public Component comp;

    @Positive
        public Card(String cardName, Component cardComponent) {
    @Positive
        }
    @Positive
    }

    @Positive
    public CardLayout() {
    @Positive
    }

    @Positive
    public CardLayout(int hgap, int vgap) {
    @Positive
    }

    @Positive
    public int getHgap();

    @Positive
    public void setHgap(int hgap);

    @Positive
    public int getVgap();

    @Positive
    public void setVgap(int vgap);

    @Positive
    public void addLayoutComponent(Component comp, Object constraints);

    @Positive
    @Deprecated
    @Positive
    public void addLayoutComponent(String name, Component comp);

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
    void checkLayout(Container parent);

    @Positive
    public void first(Container parent);

    @Positive
    public void next(Container parent);

    @Positive
    public void previous(Container parent);

    @Positive
    void showDefaultComponent(Container parent);

    @Positive
    public void last(Container parent);

    @Positive
    public void show(Container parent, String name);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0

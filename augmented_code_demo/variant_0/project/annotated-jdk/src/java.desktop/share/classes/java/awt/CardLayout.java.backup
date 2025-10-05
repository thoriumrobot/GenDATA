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
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.Serial;
import java.io.Serializable;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class CardLayout implements LayoutManager2, Serializable {

    class Card implements Serializable {

        public String name;

        public Component comp;

        public Card(String cardName, Component cardComponent) {
        }
    }

    public CardLayout() {
    }

    public CardLayout(int hgap, int vgap) {
    }

    public int getHgap();

    public void setHgap(int hgap);

    public int getVgap();

    public void setVgap(int vgap);

    public void addLayoutComponent(Component comp, Object constraints);

    @Deprecated
    public void addLayoutComponent(String name, Component comp);

    public void removeLayoutComponent(Component comp);

    public Dimension preferredLayoutSize(Container parent);

    public Dimension minimumLayoutSize(Container parent);

    public Dimension maximumLayoutSize(Container target);

    public float getLayoutAlignmentX(Container parent);

    public float getLayoutAlignmentY(Container parent);

    public void invalidateLayout(Container target);

    public void layoutContainer(Container parent);

    void checkLayout(Container parent);

    public void first(Container parent);

    public void next(Container parent);

    public void previous(Container parent);

    void showDefaultComponent(Container parent);

    public void last(Container parent);

    public void show(Container parent, String name);

    public String toString();
}

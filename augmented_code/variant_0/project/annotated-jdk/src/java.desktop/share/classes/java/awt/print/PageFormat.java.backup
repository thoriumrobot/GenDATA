/*
 * Copyright (c) 1997, 2013, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.print;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.lang.annotation.Native;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class PageFormat implements Cloneable {

    @Native
    public static final int LANDSCAPE;

    @Native
    public static final int PORTRAIT;

    @Native
    public static final int REVERSE_LANDSCAPE;

    public PageFormat() {
    }

    public Object clone();

    public double getWidth();

    public double getHeight();

    public double getImageableX();

    public double getImageableY();

    public double getImageableWidth();

    public double getImageableHeight();

    public Paper getPaper();

    public void setPaper(Paper paper);

    public void setOrientation(int orientation) throws IllegalArgumentException;

    public int getOrientation();

    public double[] getMatrix();
}

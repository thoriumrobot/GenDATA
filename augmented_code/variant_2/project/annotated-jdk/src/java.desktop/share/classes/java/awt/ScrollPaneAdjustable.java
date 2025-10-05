/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.event.AdjustmentEvent;
    @Positive
import java.awt.event.AdjustmentListener;
    @Positive
import java.awt.peer.ScrollPanePeer;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import sun.awt.AWTAccessor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class ScrollPaneAdjustable implements Adjustable, Serializable {

    @Positive
    void setSpan(int min, int max, int visible);

    @Positive
    public int getOrientation();

    @Positive
    public void setMinimum(int min);

    @Positive
    public int getMinimum();

    @Positive
    public void setMaximum(int max);

    @Positive
    public int getMaximum();

    @Positive
    public synchronized void setUnitIncrement(int u);

    @Positive
    public int getUnitIncrement();

    @Positive
    public synchronized void setBlockIncrement(int b);

    @Positive
    public int getBlockIncrement();

    @Positive
    public void setVisibleAmount(int v);

    @Positive
    public int getVisibleAmount();

    @Positive
    public void setValueIsAdjusting(boolean b);

    @Positive
    public boolean getValueIsAdjusting();

    @Positive
    public void setValue(int v);

    @Positive
    public int getValue();

    @Positive
    public synchronized void addAdjustmentListener(AdjustmentListener l);

    @Positive
    public synchronized void removeAdjustmentListener(AdjustmentListener l);

    @Positive
    public synchronized AdjustmentListener[] getAdjustmentListeners();

    @Positive
    public String toString();

    @Positive
    public String paramString();
    @Positive
}

/*
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ImageProducer;
import java.awt.image.ImageObserver;
import java.awt.image.ImageFilter;
import java.awt.image.FilteredImageSource;
import java.awt.image.AreaAveragingScaleFilter;
import java.awt.image.ReplicateScaleFilter;
import sun.awt.image.SurfaceManager;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class Image {

    protected Image() {
    }

    protected float accelerationPriority;

    public abstract int getWidth(ImageObserver observer);

    public abstract int getHeight(ImageObserver observer);

    public abstract ImageProducer getSource();

    public abstract Graphics getGraphics();

    public abstract Object getProperty(String name, ImageObserver observer);

    public static final Object UndefinedProperty;

    public Image getScaledInstance(int width, int height, int hints);

    public static final int SCALE_DEFAULT;

    public static final int SCALE_FAST;

    public static final int SCALE_SMOOTH;

    public static final int SCALE_REPLICATE;

    public static final int SCALE_AREA_AVERAGING;

    public void flush();

    public ImageCapabilities getCapabilities(GraphicsConfiguration gc);

    public void setAccelerationPriority(float priority);

    public float getAccelerationPriority();
}

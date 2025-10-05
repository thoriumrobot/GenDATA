/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ImageProducer;
    @Positive
import java.awt.image.ImageObserver;
    @Positive
import java.awt.image.ImageFilter;
    @Positive
import java.awt.image.FilteredImageSource;
    @Positive
import java.awt.image.AreaAveragingScaleFilter;
    @Positive
import java.awt.image.ReplicateScaleFilter;
    @Positive
import sun.awt.image.SurfaceManager;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Image {

    @Positive
    protected Image() {
    @Positive
    }

    @Positive
    protected float accelerationPriority;

    @Positive
    public abstract int getWidth(ImageObserver observer);

    @Positive
    public abstract int getHeight(ImageObserver observer);

    @Positive
    public abstract ImageProducer getSource();

    @Positive
    public abstract Graphics getGraphics();

    @Positive
    public abstract Object getProperty(String name, ImageObserver observer);

    @Positive
    public static final Object UndefinedProperty;

    @Positive
    public Image getScaledInstance(int width, int height, int hints);

    @Positive
    public static final int SCALE_DEFAULT;

    @Positive
    public static final int SCALE_FAST;

    @Positive
    public static final int SCALE_SMOOTH;

    @Positive
    public static final int SCALE_REPLICATE;

    @Positive
    public static final int SCALE_AREA_AVERAGING;

    @Positive
    public void flush();

    @Positive
    public ImageCapabilities getCapabilities(GraphicsConfiguration gc);

    @Positive
    public void setAccelerationPriority(float priority);

    @Positive
    public float getAccelerationPriority();
    @Positive
}

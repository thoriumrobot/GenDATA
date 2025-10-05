/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.image.BaseMultiResolutionImage;
    @Positive
import java.awt.image.BufferedImage;
    @Positive
import java.awt.image.DataBufferInt;
    @Positive
import java.awt.image.DirectColorModel;
    @Positive
import java.awt.image.MultiResolutionImage;
    @Positive
import java.awt.image.Raster;
    @Positive
import java.awt.image.WritableRaster;
    @Positive
import java.awt.peer.RobotPeer;
    @Positive
import sun.awt.AWTPermissions;
    @Positive
import sun.awt.ComponentFactory;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.image.SunWritableRaster;
    @Positive
import sun.java2d.SunGraphicsEnvironment;
    @Positive
import static sun.java2d.SunGraphicsEnvironment.toDeviceSpace;
    @Positive
import static sun.java2d.SunGraphicsEnvironment.toDeviceSpaceAbs;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Robot {

    @Positive
    public Robot() throws AWTException {
    @Positive
    }

    @Positive
    public Robot(GraphicsDevice screen) throws AWTException {
    @Positive
    }

    @Positive
    public synchronized void mouseMove(int x, int y);

    @Positive
    public synchronized void mousePress(int buttons);

    @Positive
    public synchronized void mouseRelease(int buttons);

    @Positive
    public synchronized void mouseWheel(int wheelAmt);

    @Positive
    public synchronized void keyPress(int keycode);

    @Positive
    public synchronized void keyRelease(int keycode);

    @Positive
    public synchronized Color getPixelColor(int x, int y);

    @Positive
    public synchronized BufferedImage createScreenCapture(Rectangle screenRect);

    @Positive
    public synchronized MultiResolutionImage createMultiResolutionScreenCapture(Rectangle screenRect);

    @Positive
    public synchronized boolean isAutoWaitForIdle();

    @Positive
    public synchronized void setAutoWaitForIdle(boolean isOn);

    @Positive
    public synchronized int getAutoDelay();

    @Positive
    public synchronized void setAutoDelay(int ms);

    @Positive
    public void delay(int ms);

    @Positive
    public synchronized void waitForIdle();

    @Positive
    @Override
    @Positive
    public synchronized String toString();
    @Positive
}

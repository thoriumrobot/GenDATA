/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.UI;
    @Positive
import org.checkerframework.checker.guieffect.qual.UIType;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import sun.swing.UIAction;
    @Positive
import java.applet.*;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.awt.dnd.DropTarget;
    @Positive
import java.lang.reflect.*;
    @Positive
import javax.accessibility.*;
    @Positive
import javax.swing.event.MenuDragMouseEvent;
    @Positive
import javax.swing.plaf.UIResource;
    @Positive
import javax.swing.text.View;
    @Positive
import java.security.AccessController;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTAccessor.MouseEventAccessor;

    @Positive
@AnnotatedFor({ "guieffect" })
    @Positive
@UIType
    @Positive
public class SwingUtilities implements SwingConstants {

    @Positive
    static void installSwingDropTargetAsNecessary(Component c, TransferHandler t);

    @Positive
    public static final boolean isRectangleContainingRectangle(Rectangle a, Rectangle b);

    @Positive
    public static Rectangle getLocalBounds(Component aComponent);

    @Positive
    public static Window getWindowAncestor(Component c);

    @Positive
    static Point convertScreenLocationToParent(Container parent, int x, int y);

    @Positive
    public static Point convertPoint(Component source, Point aPoint, Component destination);

    @Positive
    public static Point convertPoint(Component source, int x, int y, Component destination);

    @Positive
    public static Rectangle convertRectangle(Component source, Rectangle aRectangle, Component destination);

    @Positive
    public static Container getAncestorOfClass(Class<?> c, Component comp);

    @Positive
    public static Container getAncestorNamed(String name, Component comp);

    @Positive
    public static Component getDeepestComponentAt(Component parent, int x, int y);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static MouseEvent convertMouseEvent(Component source, MouseEvent sourceEvent, Component destination);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static void convertPointToScreen(Point p, Component c);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static void convertPointFromScreen(Point p, Component c);

    @Positive
    public static Window windowForComponent(Component c);

    @Positive
    public static boolean isDescendingFrom(Component a, Component b);

    @Positive
    public static Rectangle computeIntersection(int x, int y, int width, int height, Rectangle dest);

    @Positive
    public static Rectangle computeUnion(int x, int y, int width, int height, Rectangle dest);

    @Positive
    public static Rectangle[] computeDifference(Rectangle rectA, Rectangle rectB);

    @Positive
    public static boolean isLeftMouseButton(MouseEvent anEvent);

    @Positive
    public static boolean isMiddleMouseButton(MouseEvent anEvent);

    @Positive
    public static boolean isRightMouseButton(MouseEvent anEvent);

    @Positive
    public static int computeStringWidth(FontMetrics fm, String str);

    @Positive
    public static String layoutCompoundLabel(JComponent c, FontMetrics fm, String text, Icon icon, int verticalAlignment, int horizontalAlignment, int verticalTextPosition, int horizontalTextPosition, Rectangle viewR, Rectangle iconR, Rectangle textR, int textIconGap);

    @Positive
    public static String layoutCompoundLabel(FontMetrics fm, String text, Icon icon, int verticalAlignment, int horizontalAlignment, int verticalTextPosition, int horizontalTextPosition, Rectangle viewR, Rectangle iconR, Rectangle textR, int textIconGap);

    @Positive
    public static void paintComponent(Graphics g, Component c, Container p, int x, int y, int w, int h);

    @Positive
    public static void paintComponent(Graphics g, Component c, Container p, Rectangle r);

    @Positive
    public static void updateComponentTreeUI(Component c);

    @Positive
    @SafeEffect
    @Positive
    public static void invokeLater(@UI Runnable doRun);

    @Positive
    @SafeEffect
    @Positive
    public static void invokeAndWait(@UI final Runnable doRun) throws InterruptedException, InvocationTargetException;

    @Positive
    public static boolean isEventDispatchThread();

    @Positive
    public static int getAccessibleIndexInParent(Component c);

    @Positive
    public static Accessible getAccessibleAt(Component c, Point p);

    @Positive
    public static AccessibleStateSet getAccessibleStateSet(Component c);

    @Positive
    public static int getAccessibleChildrenCount(Component c);

    @Positive
    public static Accessible getAccessibleChild(Component c, int i);

    @Positive
    @Deprecated
    @Positive
    public static Component findFocusOwner(Component c);

    @Positive
    public static JRootPane getRootPane(Component c);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static Component getRoot(Component c);

    @Positive
    static JComponent getPaintingOrigin(JComponent c);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static boolean processKeyBindings(KeyEvent event);

    @Positive
    static boolean isValidKeyEventForKeyBindings(KeyEvent e);

    @Positive
    public static boolean notifyAction(Action action, KeyStroke ks, KeyEvent event, Object sender, int modifiers);

    @Positive
    public static void replaceUIInputMap(JComponent component, int type, InputMap uiInputMap);

    @Positive
    public static void replaceUIActionMap(JComponent component, ActionMap uiActionMap);

    @Positive
    public static InputMap getUIInputMap(JComponent component, int condition);

    @Positive
    public static ActionMap getUIActionMap(JComponent component);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class SharedOwnerFrame extends Frame implements WindowListener {

    @Positive
        public void addNotify();

    @Positive
        void installListeners();

    @Positive
        public void windowClosed(WindowEvent e);

    @Positive
        public void windowOpened(WindowEvent e);

    @Positive
        public void windowClosing(WindowEvent e);

    @Positive
        public void windowIconified(WindowEvent e);

    @Positive
        public void windowDeiconified(WindowEvent e);

    @Positive
        public void windowActivated(WindowEvent e);

    @Positive
        public void windowDeactivated(WindowEvent e);

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void show();

    @Positive
        public void dispose();
    @Positive
    }

    @Positive
    static Frame getSharedOwnerFrame() throws HeadlessException;

    @Positive
    static WindowListener getSharedOwnerFrameShutdownListener() throws HeadlessException;

    @Positive
    static Object appContextGet(Object key);

    @Positive
    static void appContextPut(Object key, Object value);

    @Positive
    static void appContextRemove(Object key);

    @Positive
    static Class<?> loadSystemClass(String className) throws ClassNotFoundException;

    @Positive
    static boolean isLeftToRight(Component c);

    @Positive
    static boolean doesIconReferenceImage(Icon icon, Image image);

    @Positive
    static int findDisplayedMnemonicIndex(String text, int mnemonic);

    @Positive
    public static Rectangle calculateInnerArea(JComponent c, Rectangle r);

    @Positive
    static void updateRendererOrEditorUI(Object rendererOrEditor);

    @Positive
    public static Container getUnwrappedParent(Component component);

    @Positive
    public static Component getUnwrappedView(JViewport viewport);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static Container getValidateRoot(Container c, boolean visibleOnly);
    @Positive
}

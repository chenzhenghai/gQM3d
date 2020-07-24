#include "Viewer.h"
#include "Mesh.h"
#include <stdio.h>

float angleY = 0;
float angleX = 0;
float farZ = -1000;
int cavityid = -1;
internalmesh* drawmesh;
bool off = false;

void init(void)
{
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1000.0f, 0.0f, 1000.0f, 0.0f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void drawPoints()
{
	if (drawmesh != NULL && drawmesh->pointlist != NULL)
	{
		double x, y, z;
		for (int i = 0; i < drawmesh->numofpoints; i++)
		{
			x = drawmesh->pointlist[3 * i];
			y = drawmesh->pointlist[3 * i + 1];
			z = drawmesh->pointlist[3 * i + 2];

			glColor3f(0.0, 0.0, 0.0);
			glBegin(GL_POINTS);
			glVertex3f(x, y, z);
			glEnd();
		}
	}
}

void drawTets()
{
	if (drawmesh != NULL && drawmesh->tetlist != NULL)
	{
		int p[4], i, j, k;
		double x[4], y[4], z[4];
		for (i = 0; i < drawmesh->numoftet; i++)
		{
			if (drawmesh->tetstatus[i].isEmpty())
				continue;

			for (j = 0; j < 4; j++)
			{
				p[j] = drawmesh->tetlist[4 * i + j];
				x[j] = drawmesh->pointlist[3 * p[j] + 0];
				y[j] = drawmesh->pointlist[3 * p[j] + 1];
				z[j] = drawmesh->pointlist[3 * p[j] + 2];
			}
			for (j = 0; j < 4; j++)
			{
				for (k = j + 1; k < 4; k++)
				{
					glColor3f(0.0, 0.0, 0.0);
					glLineWidth(0.1);
					glBegin(GL_LINES);
					glVertex3f(x[j], y[j], z[j]);
					glVertex3f(x[k], y[k], z[k]);
					glEnd();
				}
			}
		}
	}
}

void drawCavities()
{
	if (drawmesh != NULL && drawmesh->cavebdrylist != NULL)
	{
		int p[3], i, j, k, threadId;
		double x[3], y[3], z[3];
		tethandle tmp;
		for (i = 0; i < drawmesh->numofthread; i++)
		{
			threadId = drawmesh->threadlist[i];
			if (cavityid == -1 || i == cavityid)
			{
				k = drawmesh->cavebdryhead[threadId];
				while (k != -1)
				{
					tmp = drawmesh->cavebdrylist[k];
					p[0] = org(tmp, drawmesh->tetlist);
					p[1] = dest(tmp, drawmesh->tetlist);
					p[2] = apex(tmp, drawmesh->tetlist);
					if (p[0] != -1 && p[1] != -1 && p[2] != -1)
					{
						for (j = 0; j < 3; j++)
						{
							x[j] = drawmesh->pointlist[3 * p[j] + 0];
							y[j] = drawmesh->pointlist[3 * p[j] + 1];
							z[j] = drawmesh->pointlist[3 * p[j] + 2];
						}
						glColor3f(0.0f, 0.0f, 0.0);
						glBegin(GL_LINES);
						glVertex3f(x[0], y[0], z[0]);
						glVertex3f(x[1], y[1], z[1]);
						glVertex3f(x[1], y[1], z[1]);
						glVertex3f(x[2], y[2], z[2]);
						glVertex3f(x[0], y[0], z[0]);
						glVertex3f(x[2], y[2], z[2]);
						glEnd();
						glColor3f(0.0f, 0.0f, 1.0);
						glBegin(GL_TRIANGLES);
						glVertex3f(x[0], y[0], z[0]);
						glVertex3f(x[1], y[1], z[1]);
						glVertex3f(x[2], y[2], z[2]);
						glEnd();
					}

					k = drawmesh->cavebdrynext[k];
				}
			}
		}
	}
}

void drawEncroachedSubsegs()
{
	if (drawmesh != NULL && drawmesh->insertidxlist != NULL && drawmesh->insertiontype == 0)
	{
		int p[3], i, j, k, subsegid;
		double x[3], y[3], z[3];
		for (i = 0; i < drawmesh->numofinsertpt; i++)
		{
			subsegid = drawmesh->insertidxlist[i];
			for (j = 0; j < 2; j++)
			{
				p[j] = drawmesh->seglist[3 * subsegid + j];
				x[j] = drawmesh->pointlist[3 * p[j] + 0];
				y[j] = drawmesh->pointlist[3 * p[j] + 1];
				z[j] = drawmesh->pointlist[3 * p[j] + 2];
			}
			if (drawmesh->threadmarker[i] == -1)
				glColor3f(1.0f, 0.0f, 0.0);
			else
				glColor3f(0.0f, 1.0f, 0.0);
			glBegin(GL_LINES);
			glVertex3f(x[0], y[0], z[0]);
			glVertex3f(x[1], y[1], z[1]);
			glEnd();
		}
	}
}

void drawEncroachedSubfaces()
{
	if (drawmesh != NULL && drawmesh->insertidxlist != NULL && drawmesh->insertiontype == 1)
	{
		int p[3], i, j, k, subfaceid;
		double x[3], y[3], z[3];
		for (i = 0; i < drawmesh->numofinsertpt; i++)
		{
			subfaceid = drawmesh->insertidxlist[i];
			for (j = 0; j < 3; j++)
			{
				p[j] = drawmesh->trifacelist[3 * subfaceid + j];
				x[j] = drawmesh->pointlist[3 * p[j] + 0];
				y[j] = drawmesh->pointlist[3 * p[j] + 1];
				z[j] = drawmesh->pointlist[3 * p[j] + 2];
			}
			if (drawmesh->threadmarker[i] == -1)
				glColor3f(1.0f, 0.0f, 0.0);
			else
				glColor3f(0.0f, 1.0f, 0.0);
			glBegin(GL_LINES);
			glVertex3f(x[0], y[0], z[0]);
			glVertex3f(x[1], y[1], z[1]);
			glVertex3f(x[1], y[1], z[1]);
			glVertex3f(x[2], y[2], z[2]);
			glVertex3f(x[0], y[0], z[0]);
			glVertex3f(x[2], y[2], z[2]);
			glEnd();
		}
	}
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, farZ);
	
	glPushMatrix();
	glTranslatef(500.0, 500.0, 500.0);
	glRotatef(angleY, 0.0f, 1.0f, 0.0f);
	glRotatef(angleX, 1.0f, 0.0f, 0.0f);
	glTranslatef(-500.0, -500.0, -500.0);

	if (!off)
	{
		drawPoints();
		drawTets();
	}
	drawCavities();
	drawEncroachedSubsegs();
	drawEncroachedSubfaces();

	glPopMatrix();

	glFlush();
}

void reshape(int w, int h)
{
	//glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'd':angleY += 10;
		break;
	case 'a':angleY -= 10;
		break;
	case 'w':angleX -= 10;
		break;
	case 's':angleX += 10;
		break;
	case 'q': 
		farZ += 10;
		if (farZ > 0)
			farZ = 0;
		break;
	case 'e': 
		farZ -= 10;
		if (farZ < -1000)
			farZ = -1000;
		break;
	case 'r': off = !off;
		break;
	case 't':
		cavityid++;
		if (cavityid >= drawmesh->numofthread)
			cavityid = 0;
		break;
	case 'f':
		cavityid = -1;
		break;
	}
	if (angleX >= 360)
		angleX = 0;
	else if (angleX <= 0)
		angleX += 360;
	
	if (angleY >= 360)
		angleY = 0;
	else if (angleY <= 0)
		angleY += 360;
}


void drawMesh(int argc, char** argv, internalmesh* input)
{
	drawmesh = input;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(keyboard);
	glutMainLoop();
}